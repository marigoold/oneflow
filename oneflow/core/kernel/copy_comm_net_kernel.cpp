#include "oneflow/core/kernel/copy_comm_net_kernel.h"
#include "oneflow/core/comm_network/data_comm_network.h"

namespace oneflow {

void CopyCommNetKernel::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)>) const {
  auto other_val =
      static_cast<std::tuple<void**, int64_t, const void*, const void*>*>(
          kernel_ctx.other);
  void** read_id = std::get<0>(*other_val);
  int64_t src_machine_id = std::get<1>(*other_val);
  const void* readable_token = std::get<2>(*other_val);
  const void* writeable_token = std::get<3>(*other_val);
  *read_id = DataCommNet::Singleton()->Read(src_machine_id, readable_token,
                                            writeable_token);
}

COMMAND(AddKernelCreator(OperatorConf::kCopyCommNetConf,
                         []() { return new CopyCommNetKernel; }));

}  // namespace oneflow
