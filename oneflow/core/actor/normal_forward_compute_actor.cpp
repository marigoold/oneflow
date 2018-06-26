#include "oneflow/core/actor/normal_forward_compute_actor.h"

namespace oneflow {

void NormalForwardCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  model_regst_desc_id_ = Name2SoleRegstDescId("model");
  const_model_regst_desc_id_ = Name2SoleRegstDescId("const_model");
  const_buf_regst_desc_id_ = Name2SoleRegstDescId("const_buf");
  forward_model_regst_desc_id_ = Name2SoleRegstDescId("forward_model");
  random_seed_ = task_proto.random_seed();
  model_regst_ = nullptr;
  const_model_regst_ = nullptr;
  const_buf_regst_ = nullptr;
  pre_forward_model_regst_ = nullptr;
  if (forward_model_regst_desc_id_ != -1) {
    pre_forward_model_regst_ = GetCurWriteableRegst(forward_model_regst_desc_id_);
  }
  staleness_ = -1;
  if (const_buf_regst_desc_id_ != -1) {
    const_buf_regst_ = GetSoleProducedRegst(const_buf_regst_desc_id_);
  }
  if (random_seed_ == -1 || (model_regst_desc_id_ == -1 && const_model_regst_desc_id_ == -1)) {
    if (forward_model_regst_desc_id_ != -1 || const_buf_regst_desc_id_ != -1) {
      AsyncInitModelAndConstBuf();
    }
    if (forward_model_regst_desc_id_ != -1) { SendMsgToForwardModelSaveActor(0); }
    if (const_buf_regst_desc_id_ != -1) { SendConstBufInitMsgToBwActor(); }
    OF_SET_MSG_HANDLER(&NormalForwardCompActor::HandlerNormal);
  } else {
    OF_SET_MSG_HANDLER(&NormalForwardCompActor::HandlerInitModelAndConstBuf);
  }

  if (const_buf_regst_ && !const_buf_regst_->consumers_actor_id().empty()) {
    DecreaseActualWriteableProducedRegstDescNum(1);
  }
}

void NormalForwardCompActor::ForEachCurCustomizedReadableRegst(
    std::function<void(const Regst*)> handler) const {
  if (model_regst_desc_id_ != -1) { handler(model_regst_); }
  if (const_model_regst_desc_id_ != -1) { handler(const_model_regst_); }
}

void NormalForwardCompActor::NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) {
  Regst* regst = msg.regst();
  if (regst->regst_desc_id() == model_regst_desc_id_) {
    UpdateModelRegstPtr(regst);
  } else if (regst->regst_desc_id() == const_model_regst_desc_id_) {
    CHECK(const_model_regst_ == nullptr);
    const_model_regst_ = regst;
  } else {
    UNIMPLEMENTED();
  }
}

void NormalForwardCompActor::Act() {
  int64_t model_version_id = -1;
  if (model_regst_) { model_version_id = model_regst_->model_version_id(); }
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  int64_t piece_id = GetNaiveFirstCurReadable()->piece_id();
  std::tuple<int64_t, std::function<const Blob*(const LogicalBlobId&)>> other_val(
      piece_id, [this](const LogicalBlobId& lbi) -> const Blob* {
        CHECK_NOTNULL(pre_forward_model_regst_);
        return pre_forward_model_regst_->GetBlobByLbi(lbi);
      });
  kernel_ctx.other = &other_val;
  if (forward_model_regst_desc_id_ != -1) {
    pre_forward_model_regst_ = GetCurWriteableRegst(forward_model_regst_desc_id_);
  }
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
    if (regst_desc_id == model_regst_desc_id_) {
      return model_regst_;
    } else if (regst_desc_id == const_model_regst_desc_id_) {
      return const_model_regst_;
    } else if (regst_desc_id == const_buf_regst_desc_id_) {
      return const_buf_regst_;
    } else {
      return nullptr;
    }
  });
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_piece_id(piece_id);
    regst->set_model_version_id(model_version_id);
    return regst->regst_desc_id() != forward_model_regst_desc_id_;
  });
  if (Global<JobDesc>::Get()->IsTrain()) {
    if (model_regst_) {
      int64_t last_piece_id = GetLastPieceIdForModelVersionId(staleness_, model_version_id);
      CHECK_LE(piece_id, last_piece_id);
      if (piece_id == last_piece_id) { AsyncReturnModelRegst(); }
    }
    TrySendMsgToForwardModelSaveActor(piece_id);
  }
}

bool NormalForwardCompActor::IsCustomizedReadReady() {
  if (model_regst_desc_id_ != -1 && model_regst_ == nullptr) { return false; }
  if (const_model_regst_desc_id_ != -1 && const_model_regst_ == nullptr) { return false; }
  return true;
}

void NormalForwardCompActor::AsyncReturnAllCustomizedReadableRegst() {
  TryAsyncReturnModelRegst();
  TryAsyncReturnConstModelRegst();
}

int NormalForwardCompActor::HandlerInitModelAndConstBuf(const ActorMsg& msg) {
  CHECK_NE(random_seed_, -1);
  Regst* regst = msg.regst();
  if (regst->regst_desc_id() == model_regst_desc_id_) {
    model_regst_ = regst;
    CHECK_EQ(staleness_, -1);
    staleness_ = model_regst_->regst_desc()->register_num() - 1;
  } else if (regst->regst_desc_id() == const_model_regst_desc_id_) {
    const_model_regst_ = regst;
  } else {
    UNIMPLEMENTED();
  }
  if (model_regst_desc_id_ != -1 && model_regst_ == nullptr) { return 0; }
  if (const_model_regst_desc_id_ != -1 && const_model_regst_ == nullptr) { return 0; }
  AsyncInitModelAndConstBuf();
  if (model_regst_) {
    AsyncSendRegstMsgToProducer(model_regst_);
    model_regst_ = nullptr;
  }
  if (const_model_regst_) {
    AsyncSendRegstMsgToProducer(const_model_regst_);
    const_model_regst_ = nullptr;
  }
  if (forward_model_regst_desc_id_ != -1) { SendMsgToForwardModelSaveActor(0); }
  if (const_buf_regst_desc_id_ != -1) { SendConstBufInitMsgToBwActor(); }
  OF_SET_MSG_HANDLER(&NormalForwardCompActor::HandlerNormal);
  return 0;
}

void NormalForwardCompActor::UpdateModelRegstPtr(Regst* regst) {
  TryAsyncReturnModelRegst();
  model_regst_ = regst;
}

void NormalForwardCompActor::AsyncInitModelAndConstBuf() {
  for (const ExecKernel& exec_kernel : exec_kernel_vec()) {
    KernelCtx kernel_ctx = GenDefaultKernelCtx();
    std::mt19937 random_seed_gen(random_seed_);
    kernel_ctx.other = &random_seed_gen;
    exec_kernel.kernel->InitModelAndConstBuf(
        kernel_ctx, parallel_ctx(), Global<SnapshotMgr>::Get()->GetReadableSnapshot(),
        [&](const std::string& bn_in_op) {
          const LogicalBlobId& lbi = exec_kernel.kernel->BnInOp2Lbi(bn_in_op);
          Blob* blob = nullptr;
          if (model_regst_) { blob = model_regst_->GetBlobByLbi(lbi); }
          if (blob == nullptr && const_model_regst_) {
            blob = const_model_regst_->GetBlobByLbi(lbi);
          }
          if (blob == nullptr && const_buf_regst_) { blob = const_buf_regst_->GetBlobByLbi(lbi); }
          if (blob == nullptr && forward_model_regst_desc_id_ != -1) {
            blob = GetCurWriteableRegst(forward_model_regst_desc_id_)->GetBlobByLbi(lbi);
          }
          return blob;
        });
  }
}

void NormalForwardCompActor::AsyncReturnModelRegst() {
  CHECK_NOTNULL(model_regst_);
  AsyncSendRegstMsgToProducer(model_regst_);
  model_regst_ = nullptr;
}

void NormalForwardCompActor::TryAsyncReturnModelRegst() {
  if (model_regst_) { AsyncReturnModelRegst(); }
}

void NormalForwardCompActor::TryAsyncReturnConstModelRegst() {
  if (const_model_regst_) {
    AsyncSendRegstMsgToProducer(const_model_regst_);
    const_model_regst_ = nullptr;
  }
}

void NormalForwardCompActor::TrySendMsgToForwardModelSaveActor(int64_t piece_id) {
  if (forward_model_regst_desc_id_ == -1) { return; }
  bool is_last_piece_in_batch = (piece_id + 1) % Global<JobDesc>::Get()->NumOfPiecesInBatch() == 0;
  int64_t batch_id = piece_id / Global<JobDesc>::Get()->NumOfPiecesInBatch();
  if (is_last_piece_in_batch && NeedModelSave(batch_id)) {
    SendMsgToForwardModelSaveActor(batch_id);
  }
}

void NormalForwardCompActor::SendMsgToForwardModelSaveActor(int64_t batch_id) {
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_model_version_id(batch_id);
    return regst->regst_desc_id() == forward_model_regst_desc_id_;
  });
}

void NormalForwardCompActor::SendConstBufInitMsgToBwActor() {
  AsyncSendRegstMsgToConsumer(
      [&](Regst* regst) { return regst->regst_desc_id() == const_buf_regst_desc_id_; });
}

REGISTER_ACTOR(TaskType::kNormalForward, NormalForwardCompActor);
REGISTER_ACTOR(TaskType::kLoss, NormalForwardCompActor);

}  // namespace oneflow
