#ifndef ONEFLOW_CORE_OPERATOR_PRELU_OP_H_
#define ONEFLOW_CORE_OPERATOR_PRELU_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class PReluOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PReluOp);
  PReluOp() = default;
  ~PReluOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  void VirtualFixParallelDesc(ParallelDesc* pr_desc) const override;

 private:
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*) const override;

  void InferHasBatchDim(
      std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override {
    NaiveInferHasBatchDim(HasBatchDim4BnInOp);
  }

  void GetSbpSignatures(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_PRELU_OP_H_
