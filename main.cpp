#include <fcntl.h>
#include <infiniband/verbs.h>
#include <mpi.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>

using namespace std;

#define MEMLEN (1024 * 1024 * 1024)
#define InitPSN 3185

/*
 * XRC receive QPs are shareable across multiple processes. Allow any process with access to the XRC
 * domain to open an existing QP. After opening the QP, the process will receive events related to
 * the QP and be able to modify the QP.
 * So, there is no need to use this verb if we use one process per node.
 */
ibv_qp *ibv_open_qp(ibv_context *context, ibv_qp_open_attr *qp_open_attr);

struct OOBExchange {
    ibv_gid gid;
    uint16_t lid;
    uint32_t qp_num;
    uint32_t srq_num;

    explicit OOBExchange()
    {
        memset(&gid, 0, sizeof(ibv_gid));
        lid = 0;
        qp_num = 0;
        srq_num = 0;
    }
};

const int INI = 0, TGT = 1;
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Prepare for out-of-band QP info exchange
    MPI_Datatype XchgQPInfoTy;
    MPI_Type_contiguous(sizeof(OOBExchange), MPI_BYTE, &XchgQPInfoTy);
    MPI_Type_commit(&XchgQPInfoTy);

    // Init device & pd
    const char *dev_name = "mlx5_0";
    int n_devices;
    ibv_device **dev_list = ibv_get_device_list(&n_devices);
    if (!n_devices || !dev_list)
        throw std::runtime_error("cannot find any RDMA device");

    int target = -1;
    if (dev_name == nullptr)
        target = 0;
    else {
        for (int i = 0; i < n_devices; ++i)
            if (!strcmp(ibv_get_device_name(dev_list[i]), dev_name)) {
                target = i;
                break;
            }
    }
    if (target < 0)
        throw std::runtime_error("cannot find device: " + std::string(dev_name));

    ibv_context *ctx = ibv_open_device(dev_list[target]);
    ibv_free_device_list(dev_list);

    ibv_port_attr port_attr;
    ibv_gid gid;
    ibv_query_port(ctx, 1, &port_attr);
    ibv_query_gid(ctx, 1, 1, &gid);

    ibv_pd *pd = ibv_alloc_pd(ctx);

    // Init memory buffer
    char *buf = nullptr;
    if (posix_memalign(reinterpret_cast<void **>(&buf), 64, MEMLEN) != 0)
        exit(-1);
    ibv_mr *mr = ibv_reg_mr(
        pd, buf, MEMLEN, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);

    // Stick to ibv_exp_* verbs whenever possible
    if (rank == INI) {  // Initiator
        // Create CQ
        ibv_cq *cq = ibv_create_cq(ctx, 128, nullptr, nullptr, 0);
        ibv_cq *recv_cq = ibv_create_cq(ctx, 8, nullptr, nullptr, 0);  // useless placeholder

        // Create XRC INI QP
        ibv_exp_qp_init_attr qp_init_attr;
        memset(&qp_init_attr, 0, sizeof(ibv_exp_qp_init_attr));
        qp_init_attr.qp_type = IBV_QPT_XRC;
        qp_init_attr.sq_sig_all = 0;
        qp_init_attr.send_cq = cq;
        qp_init_attr.recv_cq = recv_cq;
        qp_init_attr.pd = pd;
        qp_init_attr.comp_mask = IBV_EXP_QP_INIT_ATTR_PD;
        qp_init_attr.cap.max_send_wr = 128;
        qp_init_attr.cap.max_recv_wr = 128;
        qp_init_attr.cap.max_send_sge = 4;
        qp_init_attr.cap.max_recv_sge = 4;
        ibv_qp *qp = ibv_exp_create_qp(ctx, &qp_init_attr);
        if (!qp) {
            perror("[INI] create qp");
            exit(-1);
        }

        // ============================ SYNC BEGIN ============================
        OOBExchange xchg, tgt_xchg;
        xchg.gid = gid;
        xchg.lid = port_attr.lid;
        xchg.qp_num = qp->qp_num;

        MPI_Status mpirc;
        MPI_Send(&xchg, 1, XchgQPInfoTy, TGT, 0, MPI_COMM_WORLD);  // Send to TGT first
        MPI_Recv(&tgt_xchg, 1, XchgQPInfoTy, TGT, 0, MPI_COMM_WORLD,
                 &mpirc);  // Recv from TGT then
        // ============================  SYNC END  ============================

        // Modify QP to INIT -> RTR -> RTS
        ibv_qp_attr qp_attr;
        int rc = 0;

        // RESET -> INIT
        memset(&qp_attr, 0, sizeof(ibv_qp_attr));
        qp_attr.qp_state = IBV_QPS_INIT;
        qp_attr.port_num = 1;
        qp_attr.pkey_index = 0;
        qp_attr.qp_access_flags =
            IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
        rc = ibv_modify_qp(qp, &qp_attr,
                           IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
        if (rc) {
            perror("[INI] qp reset -> init");
            exit(-1);
        }

        // INIT -> RTR
        memset(&qp_attr, 0, sizeof(ibv_qp_attr));
        qp_attr.qp_state = IBV_QPS_RTR;
        qp_attr.path_mtu = IBV_MTU_4096;
        qp_attr.dest_qp_num = tgt_xchg.qp_num;  // Remote XRC TGT QP num
        qp_attr.rq_psn = InitPSN;

        qp_attr.ah_attr.dlid = tgt_xchg.lid;
        qp_attr.ah_attr.sl = 0;
        qp_attr.ah_attr.src_path_bits = 0;
        qp_attr.ah_attr.port_num = 1;
        qp_attr.ah_attr.is_global = 1;
        memcpy(&qp_attr.ah_attr.grh.dgid, &tgt_xchg.gid, sizeof(ibv_gid));
        qp_attr.ah_attr.grh.flow_label = 0;
        qp_attr.ah_attr.grh.hop_limit = 1;
        qp_attr.ah_attr.grh.sgid_index = 1;
        qp_attr.ah_attr.grh.traffic_class = 0;
        qp_attr.max_dest_rd_atomic = 16;
        qp_attr.min_rnr_timer = 12;
        ibv_modify_qp(qp, &qp_attr,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
        if (rc) {
            perror("[INI] qp init -> rtr");
            exit(-1);
        }

        // RTR -> RTS
        memset(&qp_attr, 0, sizeof(ibv_qp_attr));
        qp_attr.qp_state = IBV_QPS_RTS;
        qp_attr.sq_psn = InitPSN;
        qp_attr.timeout = 14;
        qp_attr.retry_cnt = 7;
        qp_attr.rnr_retry = 7;
        qp_attr.max_rd_atomic = 16;
        ibv_modify_qp(qp, &qp_attr,
                      IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                          IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC);
        if (rc) {
            perror("[INI] qp rtr -> rts");
            exit(-1);
        }

        // Sync with receiver side
        MPI_Barrier(MPI_COMM_WORLD);

        // Perform send
        int nchars = sprintf(buf, "Hello RDMA XRC!");

        ibv_exp_send_wr wr, *bad_wr;
        ibv_sge sge;
        sge.addr = reinterpret_cast<uintptr_t>(buf);
        sge.length = nchars;
        sge.lkey = mr->lkey;

        memset(&wr, 0, sizeof(ibv_exp_send_wr));
        wr.next = nullptr;
        wr.wr_id = 0;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.exp_opcode = IBV_EXP_WR_SEND;
        wr.exp_send_flags = IBV_EXP_SEND_SIGNALED;
        wr.xrc_remote_srq_num = tgt_xchg.srq_num;  // Remote SRQ num
        ibv_exp_post_send(qp, &wr, &bad_wr);

        ibv_wc wc[2];
        memset(wc, 0, sizeof(ibv_wc) * 2);
        while (ibv_poll_cq(cq, 1, wc) == 0)
            ;
        if (wc[0].status != IBV_WC_SUCCESS)
            fprintf(stderr, "[INI] send wc error %d\n", wc[0].status);

        // Sync with receiver side (now he should printed what I sent to him!)
        MPI_Barrier(MPI_COMM_WORLD);

        // Initiator clean-up
        ibv_destroy_qp(qp);
        ibv_destroy_cq(cq);
        ibv_destroy_cq(recv_cq);
    }
    else {  // Target
        // Open XRC Domain
        ibv_xrcd_init_attr xrcd_init_attr;
        memset(&xrcd_init_attr, 0, sizeof(ibv_xrcd_init_attr));
        xrcd_init_attr.fd = -1;
        xrcd_init_attr.oflags = O_CREAT;
        xrcd_init_attr.comp_mask = IBV_XRCD_INIT_ATTR_FD | IBV_XRCD_INIT_ATTR_OFLAGS;
        ibv_xrcd *xrcd = ibv_open_xrcd(ctx, &xrcd_init_attr);
        if (!xrcd) {
            perror("[TGT] open xrcd");
            exit(-1);
        }

        // Create CQ
        ibv_cq *send_cq = ibv_create_cq(ctx, 8, nullptr, nullptr, 0);  // useless placeholder
        ibv_cq *cq = ibv_create_cq(ctx, 128, nullptr, nullptr, 0);

        // Create SRQ
        ibv_exp_create_srq_attr srq_init_attr;
        memset(&srq_init_attr, 0, sizeof(ibv_exp_create_srq_attr));
        srq_init_attr.pd = pd;
        srq_init_attr.xrcd = xrcd;
        srq_init_attr.cq = cq;
        srq_init_attr.srq_type = IBV_EXP_SRQT_XRC;
        srq_init_attr.base.attr.max_sge = 4;
        srq_init_attr.base.attr.max_wr = 128;
        /*
            The value that the SRQ will be armed with. When the number of outstanding WRs in the SRQ
           drops below this limit, the affiliated asynchronous event IBV_EVENT_SRQ_LIMIT_REACHED
           will be generated. Value can be [0..number of WR that can be posted to the SRQ]. 0 means
           that the SRQ limit event won’t be generated (since the number of outstanding WRs in the
           SRQ can’t be negative).
         */
        srq_init_attr.base.attr.srq_limit = 0;
        srq_init_attr.comp_mask = IBV_EXP_CREATE_SRQ_CQ | IBV_EXP_CREATE_SRQ_XRCD;
        ibv_srq *srq = ibv_exp_create_srq(ctx, &srq_init_attr);

        uint32_t srq_num;
        ibv_get_srq_num(srq, &srq_num);

        // Can use ibv_exp_create_qp and related structs? to be examined
        // Create XRC TGT QP
        ibv_exp_qp_init_attr qp_init_attr;
        memset(&qp_init_attr, 0, sizeof(ibv_exp_qp_init_attr));
        qp_init_attr.qp_type = IBV_QPT_XRC_RECV;
        qp_init_attr.sq_sig_all = 0;
        qp_init_attr.send_cq = send_cq;
        qp_init_attr.recv_cq = cq;
        qp_init_attr.pd = pd;
        qp_init_attr.xrcd = xrcd;
        qp_init_attr.srq = srq;
        qp_init_attr.comp_mask = IBV_EXP_QP_INIT_ATTR_PD | IBV_EXP_QP_INIT_ATTR_XRCD;
        qp_init_attr.cap.max_send_wr = 128;
        qp_init_attr.cap.max_recv_wr = 128;
        qp_init_attr.cap.max_send_sge = 4;
        qp_init_attr.cap.max_recv_sge = 4;
        ibv_qp *qp = ibv_exp_create_qp(ctx, &qp_init_attr);
        if (!qp) {
            perror("[TGT] create qp");
            exit(-1);
        }

        // ============================ SYNC BEGIN ============================
        OOBExchange xchg, ini_xchg;
        xchg.gid = gid;
        xchg.lid = port_attr.lid;
        xchg.qp_num = qp->qp_num;
        xchg.srq_num = srq_num;

        MPI_Status mpirc;
        MPI_Recv(&ini_xchg, 1, XchgQPInfoTy, INI, 0, MPI_COMM_WORLD,
                 &mpirc);                                          // Recv from INI first
        MPI_Send(&xchg, 1, XchgQPInfoTy, INI, 0, MPI_COMM_WORLD);  // Send to INI then
        // ============================  SYNC END  ============================

        // Modify QP to INIT -> RTR -> RTS (specified by Annex A.14)
        ibv_qp_attr qp_attr;
        int rc = 0;

        // RESET -> INIT
        memset(&qp_attr, 0, sizeof(ibv_qp_attr));
        qp_attr.qp_state = IBV_QPS_INIT;
        qp_attr.port_num = 1;
        qp_attr.pkey_index = 0;
        qp_attr.qp_access_flags =
            IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
        rc = ibv_modify_qp(qp, &qp_attr,
                           IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
        if (rc) {
            perror("[TGT] qp reset -> init");
            exit(-1);
        }

        // INIT -> RTR
        memset(&qp_attr, 0, sizeof(ibv_qp_attr));
        qp_attr.qp_state = IBV_QPS_RTR;
        qp_attr.path_mtu = IBV_MTU_4096;
        qp_attr.dest_qp_num = ini_xchg.qp_num;  // ????
        qp_attr.rq_psn = InitPSN;

        qp_attr.ah_attr.dlid = ini_xchg.lid;
        qp_attr.ah_attr.sl = 0;
        qp_attr.ah_attr.src_path_bits = 0;
        qp_attr.ah_attr.port_num = 1;
        qp_attr.ah_attr.is_global = 1;
        memcpy(&qp_attr.ah_attr.grh.dgid, &ini_xchg.gid, sizeof(ibv_gid));
        qp_attr.ah_attr.grh.flow_label = 0;
        qp_attr.ah_attr.grh.hop_limit = 1;
        qp_attr.ah_attr.grh.sgid_index = 1;
        qp_attr.ah_attr.grh.traffic_class = 0;
        qp_attr.max_dest_rd_atomic = 16;
        qp_attr.min_rnr_timer = 12;
        rc = ibv_modify_qp(qp, &qp_attr,
                           IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                               IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
        if (rc) {
            perror("[TGT] qp init -> rtr");
            exit(-1);
        }

        // RTR -> RTS
        memset(&qp_attr, 0, sizeof(ibv_qp_attr));
        qp_attr.qp_state = IBV_QPS_RTS;
        qp_attr.sq_psn = InitPSN;
        qp_attr.timeout = 14;
        qp_attr.retry_cnt = 7;
        qp_attr.rnr_retry = 7;
        qp_attr.max_rd_atomic = 16;
        rc = ibv_modify_qp(qp, &qp_attr,
                           IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                               IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC);
        if (rc) {
            perror("[TGT] qp rtr -> rts");
            exit(-1);
        }

        // Sync with requester side
        MPI_Barrier(MPI_COMM_WORLD);

        // Perform recv
        memset(buf, 0, MEMLEN);

        ibv_recv_wr wr, *bad_wr;
        ibv_sge sge;
        sge.addr = reinterpret_cast<uintptr_t>(buf);
        sge.length = 128;
        sge.lkey = mr->lkey;

        memset(&wr, 0, sizeof(wr));
        wr.next = nullptr;
        wr.wr_id = 0;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        ibv_post_srq_recv(srq, &wr, &bad_wr);

        ibv_wc wc[2];
        memset(wc, 0, sizeof(ibv_wc) * 2);
        while (ibv_poll_cq(cq, 1, wc) == 0)
            ;
        if (wc[0].status != IBV_WC_SUCCESS)
            fprintf(stderr, "[TGT] recv wc error %d\n", wc[0].status);

        // Print what I received and sync
        fprintf(stderr, "[TGT] received = %s\n", buf);
        MPI_Barrier(MPI_COMM_WORLD);

        // Target clean-up
        ibv_destroy_qp(qp);
        ibv_destroy_srq(srq);
        ibv_destroy_cq(cq);
        ibv_destroy_cq(send_cq);

        // Close XRC domain
        ibv_close_xrcd(xrcd);
    }

    // Free memory buffer
    ibv_dereg_mr(mr);
    free(buf);

    // Free device & pd
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);

    MPI_Finalize();
    return 0;
}