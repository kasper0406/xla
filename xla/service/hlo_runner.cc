/* Copyright 2017 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/hlo_runner.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/literal.h"
#include "xla/service/backend.h"
#include "xla/service/computation_layout.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/hlo_module_util.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {

namespace {
class HloRunnerExecutable : public OpaqueExecutable {
 public:
  HloRunnerExecutable(const HloRunner* absl_nonnull creator,
                      std::unique_ptr<Executable> executable)
      : OpaqueExecutable(creator), executable_(std::move(executable)) {}

  Executable* executable() const { return executable_.get(); }
  std::unique_ptr<Executable> MoveExecutable() {
    return std::move(executable_);
  }

  static absl::StatusOr<HloRunnerExecutable*> TryUnwrap(
      const HloRunner& runner, OpaqueExecutable* absl_nonnull const wrapped) {
    return OpaqueExecutable::TryUnwrap<HloRunnerExecutable>(runner, wrapped);
  }
  static absl::StatusOr<const HloRunnerExecutable*> TryUnwrap(
      const HloRunner& runner,
      const OpaqueExecutable* absl_nonnull const wrapped) {
    return OpaqueExecutable::TryUnwrap<HloRunnerExecutable>(runner, wrapped);
  }

 private:
  std::unique_ptr<Executable> executable_;
};
}  // namespace

HloRunner::HloRunner(se::Platform* platform, int intra_op_parallelism_threads) {
  BackendOptions backend_options;
  backend_options.set_platform(platform);
  backend_options.set_intra_op_parallelism_threads(
      intra_op_parallelism_threads);
  backend_ = Backend::CreateBackend(backend_options).value();
  device_shape_representation_fn_ = [this](const Shape& shape) {
    return backend_->compiler()->DefaultDeviceShapeRepresentation(shape);
  };
  VLOG(1) << "Created HloRunner for platform: " << platform->Name();
}

HloRunner::~HloRunner() {}

se::DeviceMemoryAllocator* HloRunner::GetAllocator() {
  absl::MutexLock lock(&mu_);
  if (allocator_ == nullptr) {
    allocator_ = std::make_unique<se::StreamExecutorMemoryAllocator>(
        backend().default_stream_executor());
  }
  return allocator_.get();
}

absl::StatusOr<ScopedShapedBuffer> HloRunner::TransferLiteralToDevice(
    const Literal& literal,
    const ComputationLayout* absl_nullable entry_computation_layout,
    int64_t param_no,
    int64_t device_ordinal) {
  auto shape_representation_fn = [this, entry_computation_layout,
                                  param_no](const Shape& shape) {
    Shape new_shape = device_shape_representation_fn_(shape);
    if (entry_computation_layout == nullptr) {
      return new_shape;
    }

    Shape entry_computation_shape =
        entry_computation_layout->parameter_shape(param_no);
    // Favor entry computation shape with some adjustment.
    ShapeUtil::ForEachMutableSubshape(
        &new_shape,
        [&entry_computation_shape](Shape* subshape, const ShapeIndex& index) {
          if (!subshape->IsArray()) {
            return;
          }
          Shape entry_computation_subshape =
              ShapeUtil::GetSubshape(entry_computation_shape, index);
          if (entry_computation_subshape.is_static() &&
              !entry_computation_subshape.layout().tiles().empty() &&
              *subshape != entry_computation_subshape) {
            *subshape = entry_computation_subshape;
          }
        });
    return new_shape;
  };

  VLOG(2) << "Before getting raw stream for device ordinal: " << device_ordinal;
  TF_ASSIGN_OR_RETURN(
    auto raw_stream,
    backend().stream_executor(device_ordinal));
  // se::DeviceMemoryAllocator* memory_allocator = (se::DeviceMemoryAllocator*)raw_stream;

  VLOG(2) << "Before allocating for device ordinal: " << device_ordinal;
  TF_ASSIGN_OR_RETURN(
      ScopedShapedBuffer buffer,
      backend().transfer_manager()->AllocateScopedShapedBuffer(
          literal.shape(), backend().memory_allocator(),
          device_ordinal, shape_representation_fn));
  VLOG(2) << "After allocating for device ordinal: " << device_ordinal;
  TF_ASSIGN_OR_RETURN(
      auto stream, backend().BorrowStream(raw_stream));
  VLOG(2) << "After borrowing stream for device ordinal: " << device_ordinal;
  TF_RETURN_IF_ERROR(backend().transfer_manager()->TransferLiteralToDevice(
      stream.get(), literal, buffer));
  return std::move(buffer);
}

absl::StatusOr<std::vector<std::vector<ScopedShapedBuffer>>>
HloRunner::TransferLiteralsToDevices(
    absl::Span<const Literal* const> literals,
    const ComputationLayout* absl_nullable entry_computation_layout) {
  std::vector<std::vector<ScopedShapedBuffer>> all_buffers;
  for (uint64_t device_ordinal = 0; device_ordinal < backend().device_count(); ++device_ordinal) {
    VLOG(2) << "Transferring literals to device ordinal: "
            << device_ordinal;
    std::vector<ScopedShapedBuffer> buffers;
    buffers.reserve(literals.size());
    for (auto i = 0; i < literals.size(); i++) {
      const Literal* literal = literals[i];
      CHECK(literal != nullptr);
      TF_ASSIGN_OR_RETURN(
          ScopedShapedBuffer buffer,
          TransferLiteralToDevice(*literal, entry_computation_layout, i, device_ordinal));
      buffers.push_back(std::move(buffer));
    }
    all_buffers.push_back(std::move(buffers));
  }
  return std::move(all_buffers);
}

absl::StatusOr<std::vector<std::vector<ScopedShapedBuffer>>>
HloRunner::TransferLiteralsToDevices(absl::Span<const Literal> literals) {
  std::vector<const Literal*> literal_pointers;
  literal_pointers.reserve(literals.size());
  for (const auto& literal : literals) {
    literal_pointers.push_back(&literal);
  }
  return TransferLiteralsToDevices(literal_pointers, nullptr);
}

absl::StatusOr<Literal> HloRunner::TransferLiteralFromDevice(
    const ShapedBuffer& buffer) {
  TF_ASSIGN_OR_RETURN(
      auto stream, backend().BorrowStream(backend().default_stream_executor()));

  if (buffer.on_device_shape().is_static()) {
    return backend().transfer_manager()->TransferLiteralFromDevice(stream.get(),
                                                                   buffer);
  }

  Shape device_shape = buffer.on_device_shape();
  // Read real literal's shape first.
  TF_RETURN_IF_ERROR(backend().transfer_manager()->ReadDynamicShapes(
      stream.get(), &buffer, &device_shape));

  ShapedBuffer shaped_buffer(device_shape, buffer.device_ordinal());
  // Populate buffer element by element since the shapes differ now.
  shaped_buffer.buffers().ForEachMutableElement(
      [&](const xla::ShapeIndex& index, se::DeviceMemoryBase* base_buffer) {
        *base_buffer = buffer.buffer(index);
      });
  return backend().transfer_manager()->TransferLiteralFromDevice(stream.get(),
                                                                 shaped_buffer);
}

absl::StatusOr<Literal> HloRunner::Execute(
    std::unique_ptr<HloModule> module,
    absl::Span<const Literal* const> arguments, bool run_hlo_passes,
    ExecutionProfile* profile) {
  MaybeUpdateEntryComputationLayout(module.get());
  TF_ASSIGN_OR_RETURN(
      std::vector<std::vector<ScopedShapedBuffer>> argument_buffers,
      TransferLiteralsToDevices(arguments, &module->entry_computation_layout()));
  TF_ASSIGN_OR_RETURN(ExecutionOutput result,
                      ExecuteWithMovedDeviceBuffersAndBufferAssignment(
                          /*module=*/std::move(module),
                          /*buffer_assignment_proto=*/nullptr,
                          /*arguments=*/std::move(argument_buffers),
                          /*run_hlo_passes=*/run_hlo_passes,
                          /*profile=*/profile));
  return TransferLiteralFromDevice(result.Result());
}

absl::StatusOr<Literal> HloRunner::ExecuteWithBufferAssignment(
    std::unique_ptr<HloModule> module,
    const BufferAssignmentProto* buffer_assignment_proto,
    absl::Span<const Literal* const> arguments, bool run_hlo_passes,
    ExecutionProfile* profile) {
  MaybeUpdateEntryComputationLayout(module.get());
  TF_ASSIGN_OR_RETURN(
      std::vector<std::vector<ScopedShapedBuffer>> argument_buffers,
      TransferLiteralsToDevices(arguments, &module->entry_computation_layout()));
  TF_ASSIGN_OR_RETURN(ExecutionOutput result,
                      ExecuteWithMovedDeviceBuffersAndBufferAssignment(
                          /*module=*/std::move(module), buffer_assignment_proto,
                          /*arguments=*/std::move(argument_buffers),
                          /*run_hlo_passes=*/run_hlo_passes,
                          /*profile=*/profile));
  return TransferLiteralFromDevice(result.Result());
}

absl::StatusOr<Literal> HloRunner::ExecuteWithExecutable(
    OpaqueExecutable* executable, absl::Span<const Literal* const> arguments,
    ExecutionProfile* profile) {
  TF_ASSIGN_OR_RETURN(HloRunnerExecutable* const hlo_runner_executable,
                      HloRunnerExecutable::TryUnwrap(*this, executable));
  TF_ASSIGN_OR_RETURN(
      std::vector<std::vector<ScopedShapedBuffer>> argument_buffers,
      TransferLiteralsToDevices(arguments, &hlo_runner_executable->executable()
                                               ->module()
                                               .entry_computation_layout()));
  TF_ASSIGN_OR_RETURN(ExecutionOutput result,
                      ExecuteWithDeviceBuffers(
                          /*executable=*/hlo_runner_executable,
                          /*arguments=*/std::move(argument_buffers),
                          /*profile=*/profile));
  return TransferLiteralFromDevice(result.Result());
}

// Create a partially owning vector of `ExecutionInput`s based on an owning
// vector of `OwningDeviceMemory`'s.
//
// This function creates owning references to memory which is already
// owned by a ScopedShapedBuffer. This can result in double-free and similar
// problems in rare cases (for example when the running of the HLO is
// unsuccessful). We keep this here because too much code depends on it for
// repeatedly running HLOs without reallocating device buffers.
static std::vector<ExecutionInput> ExecutionInputsFromScopedShapedBuffers(
    absl::Span<ScopedShapedBuffer const> inputs,
    HloInputOutputAliasConfig alias_config, int device_ordinal,
    se::DeviceMemoryAllocator* allocator) {
  std::vector<ExecutionInput> execution_inputs;

  for (int param_num = 0; param_num < inputs.size(); param_num++) {
    const ScopedShapedBuffer& input_buffer = inputs[param_num];
    ShapeTree<MaybeOwningDeviceMemory> buffer_tree(
        input_buffer.on_device_shape());

    input_buffer.buffers().ForEachElement(
        [&](const ShapeIndex& index,
            const se::DeviceMemoryBase& execution_input_buffer) {
          if (alias_config.ParameterHasAlias(param_num, index)) {
            // Store owned.
            *buffer_tree.mutable_element(index) = se::OwningDeviceMemory{
                execution_input_buffer, device_ordinal, allocator};
          } else {
            // Store unowned.
            *buffer_tree.mutable_element(index) = execution_input_buffer;
          }
        });
    execution_inputs.emplace_back(std::move(buffer_tree));
  }
  return execution_inputs;
}

// Convert the owning buffer of inputs into a (partially) owning vector of
// ExecutionInputs, and an owning vector of `OwningDeviceMemory`'s.
static void ExecutionInputsFromMovedScopedShapedBuffers(
    std::vector<ExecutionInput>* out_execution_inputs,
    std::vector<se::OwningDeviceMemory>* out_owned_args,
    std::vector<ScopedShapedBuffer> inputs,
    HloInputOutputAliasConfig alias_config, int device_ordinal,
    se::DeviceMemoryAllocator* allocator) {
  CHECK(out_execution_inputs->empty());
  CHECK(out_owned_args->empty());

  for (int param_num = 0; param_num < inputs.size(); param_num++) {
    ShapedBuffer input_buffer = inputs[param_num].release();

    ShapeTree<MaybeOwningDeviceMemory> buffer_tree(
        input_buffer.on_device_shape());
    
    VLOG(2) << "On host shape: " << input_buffer.on_host_shape().ToString();
    VLOG(2) << "On device shape: " << input_buffer.on_device_shape().ToString();
    VLOG(2) << "Num buffer leafs: " << input_buffer.buffers().leaf_count();

    input_buffer.buffers().ForEachElement(
        [&](const ShapeIndex& index,
            const se::DeviceMemoryBase& execution_input_buffer) {
          if (alias_config.ParameterHasAlias(param_num, index)) {
            VLOG(1) << "Input " << param_num << " index " << index.ToString()
                    << " buffer " << execution_input_buffer.opaque()
                    << " will be owned by out_execution_inputs.";

            // Owned by out_execution_inputs.
            // This allows the Executable to transfer the ownership to the
            // ExecutionOutput.
            *buffer_tree.mutable_element(index) = se::OwningDeviceMemory{
                execution_input_buffer, device_ordinal, allocator};
          } else {
            VLOG(1) << "Input " << param_num << " index " << index.ToString()
                    << " buffer " << execution_input_buffer.opaque()
                    << " will be owned by out_owned_args.";

            // Not owned by out_execution_inputs.
            *buffer_tree.mutable_element(index) = execution_input_buffer;
            // Owned by out_owned_args.
            out_owned_args->emplace_back(execution_input_buffer, device_ordinal,
                                         allocator);
          }
        });
    out_execution_inputs->emplace_back(std::move(buffer_tree));
  }
}

absl::StatusOr<ExecutionOutput> HloRunner::ExecuteWithDeviceBuffers(
    std::unique_ptr<HloModule> module,
    std::vector<std::vector<ScopedShapedBuffer>> arguments, bool run_hlo_passes,
    ExecutionProfile* profile) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<OpaqueExecutable> executable,
                      CreateExecutable(std::move(module), run_hlo_passes));
  TF_ASSIGN_OR_RETURN(HloRunnerExecutable* const hlo_runner_executable,
                      HloRunnerExecutable::TryUnwrap(*this, executable.get()));
  return ExecuteWithDeviceBuffers(hlo_runner_executable, std::move(arguments), profile);
}

absl::StatusOr<ExecutionOutput> HloRunner::ExecuteWithDeviceBuffers(
    OpaqueExecutable* executable,
    std::vector<std::vector<ScopedShapedBuffer>> all_arguments, ExecutionProfile* profile) {
  TF_ASSIGN_OR_RETURN(HloRunnerExecutable* const hlo_runner_executable,
                      HloRunnerExecutable::TryUnwrap(*this, executable));
  
  std::vector<std::vector<ExecutionInput>> all_execution_arguments;
  for (int64_t device_ordinal = 0;
       device_ordinal < backend().device_count(); ++device_ordinal) {
    TF_ASSIGN_OR_RETURN(
      auto raw_stream,
      backend().stream_executor(device_ordinal));
    se::DeviceMemoryAllocator* memory_allocator = (se::DeviceMemoryAllocator*)raw_stream;

    std::vector<ExecutionInput> execution_arguments = ExecutionInputsFromScopedShapedBuffers(
        all_arguments[device_ordinal],
        hlo_runner_executable->executable()
            ->module()
            .input_output_alias_config(),
        device_ordinal,
        memory_allocator);
    all_execution_arguments.push_back(std::move(execution_arguments));
  }
  
  return ExecuteWithExecutionInputs(hlo_runner_executable->executable(),
                                    std::move(all_execution_arguments), profile);
}

absl::StatusOr<ExecutionOutput> HloRunner::ExecuteWithMovedDeviceBuffers(
    std::unique_ptr<HloModule> module,
    std::vector<std::vector<ScopedShapedBuffer>> arguments, bool run_hlo_passes,
    ExecutionProfile* profile) {
  return ExecuteWithMovedDeviceBuffersAndBufferAssignment(
      std::move(module), /*buffer_assignment_proto=*/nullptr,
      std::move(arguments), run_hlo_passes, profile);
}

absl::StatusOr<ExecutionOutput>
HloRunner::ExecuteWithMovedDeviceBuffersAndBufferAssignment(
    std::unique_ptr<HloModule> module,
    const BufferAssignmentProto* buffer_assignment_proto,
    std::vector<std::vector<ScopedShapedBuffer>> arguments, bool run_hlo_passes,
    ExecutionProfile* profile) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<OpaqueExecutable> executable,
      CreateExecutableWithBufferAssignment(
          std::move(module), buffer_assignment_proto, run_hlo_passes));
  TF_ASSIGN_OR_RETURN(HloRunnerExecutable* const hlo_runner_executable,
                      HloRunnerExecutable::TryUnwrap(*this, executable.get()));
  return ExecuteWithMovedDeviceBuffers(hlo_runner_executable->executable(),
                                       std::move(arguments), profile);
}

absl::StatusOr<ExecutionOutput> HloRunner::ExecuteWithMovedDeviceBuffers(
    Executable* executable, std::vector<std::vector<ScopedShapedBuffer>> all_arguments,
    ExecutionProfile* profile) {
  std::vector<std::vector<ExecutionInput>> all_execution_arguments;
  // We need this to keep the arguments not owned by execution_arguments
  // alive.
  std::vector<std::vector<se::OwningDeviceMemory>> all_owned_arguments;

  for (int64_t device = 0; device < all_arguments.size(); ++device) {
    std::vector<ExecutionInput> execution_arguments;
    // We need this to keep the arguments not owned by execution_arguments
    // alive.
    std::vector<se::OwningDeviceMemory> owned_arguments;

    TF_ASSIGN_OR_RETURN(
      auto stream_executor,
      backend().stream_executor(device));
    se::DeviceMemoryAllocator* allocator = (se::DeviceMemoryAllocator*)stream_executor;

    ExecutionInputsFromMovedScopedShapedBuffers(
        &execution_arguments, &owned_arguments, std::move(all_arguments[device]),
        executable->module().input_output_alias_config(),
        device, allocator);
    
    all_execution_arguments.push_back(std::move(execution_arguments));
    all_owned_arguments.push_back(std::move(owned_arguments));
  }

  TF_ASSIGN_OR_RETURN(ExecutionOutput retval,
                      ExecuteWithExecutionInputs(
                          executable, std::move(all_execution_arguments), profile));

  // This is here to make sure that the output buffers get freed up when the
  // ExecutionOutput is destroyed.
  retval.Commit();
  return retval;
}

absl::StatusOr<ExecutionOutput> HloRunner::ExecuteWithExecutionInputs(
    Executable* executable, std::vector<std::vector<ExecutionInput>> arguments,
    ExecutionProfile* profile) {
  MaybeUpdateEntryComputationLayout(&executable->module());

  VLOG(2) << "Running with nullptr device assignment";
  VLOG(2) << "Module config num_replicas: "
          << executable->module_config().replica_count();
  VLOG(2) << "Module config num_partitions: "
          << executable->module_config().num_partitions();
  TF_ASSIGN_OR_RETURN(auto device_assignment, backend().computation_placer()->AssignDevices(
    executable->module_config().replica_count(),
    executable->module_config().num_partitions()));
  VLOG(2) << "Arguments size: " << arguments.size();


  if (backend().device_count() > 1) {
    std::vector<std::unique_ptr<se::Stream>> streams;
    for (int64_t device = 0; device < backend().device_count(); ++device) {
      TF_ASSIGN_OR_RETURN(
          se::StreamExecutor* stream_executor,
          backend().stream_executor(device));
      TF_ASSIGN_OR_RETURN(auto stream,
                          stream_executor->CreateStream());
      streams.push_back(std::move(stream));
    }

    absl::Mutex mutex;
    std::vector<absl::StatusOr<ExecutionOutput>> thread_results(
            backend().device_count());

    { // Scope for the thread pool.
      std::unique_ptr<tsl::thread::ThreadPool> pool = std::make_unique<tsl::thread::ThreadPool>(
            tsl::Env::Default(), "execution_pool",
            /*num_threads=*/backend().device_count());

      auto run_id = RunId();
      for (int64_t device = 0; device < backend().device_count(); ++device) {
        VLOG(2) << "Starting execution on device: " << device;
        auto stream = streams[device].get();
        auto device_arguments = std::move(arguments[device]);
        auto args_ptr = std::make_unique<std::vector<xla::ExecutionInput>>(std::move(device_arguments));

        auto core_task = [this, run_id, &thread_results, device, stream, &device_assignment, &profile, &executable, moved_args = std::move(args_ptr), &mutex]() mutable {
          VLOG(2) << "Executing on device has started: " << device;
          // Get service run options. 
          ServiceExecutableRunOptions service_run_options =
              GetServiceRunOptionsForDevice(device,
                                            stream, &device_assignment, run_id,
                                            backend().device_count());
          service_run_options.mutable_run_options()->set_execution_profile(profile);

          VLOG(2) << "Created service run options for device: " << device;

          auto options = executable->module().config().debug_options();
          auto gpu_run_options = std::make_unique<gpu::GpuExecutableRunOptions>();
          if (options.xla_gpu_require_exclusive_lock()) {
            gpu_run_options->set_requires_exclusive_lock_on_gpu();
          }
          
          VLOG(2) << "Set GPU run options for device: " << device;

          service_run_options.mutable_run_options()->set_gpu_executable_run_options(
              gpu_run_options.get());

          VLOG(2) << "Executing on device: " << device;

          auto retval = executable->ExecuteOnStreamWrapper(&service_run_options, std::move(*moved_args));
          TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
          // TF_ASSIGN_OR_RETURN(Literal literal,
          //                     backend().transfer_manager()->TransferLiteralFromDevice(
          //                         streams[i].get(), results[i]));
          // exec_results.push_back(std::move(literal));

          absl::MutexLock lock(&mutex);
          thread_results[device] = std::move(retval);
          return absl::OkStatus();
        };

        auto shared_task_ptr = std::make_shared<decltype(core_task)>(std::move(core_task));
        pool->Schedule([shared_task_ptr]() { 
            (*shared_task_ptr)(); // Call the core task.
        });
      }
    } // End of thread pool scope. Execution will block until all threads finish.

    VLOG(2) << "Threads finished execution";
    for (auto& thread_result : thread_results) {
      if (!thread_result.ok()) {
        VLOG(2) << "Thread execution failed: " << thread_result.status();
        return thread_result.status();
      }
      // results.push_back(std::move(thread_result).value());
      return std::move(thread_result).value();
    }
    return absl::InternalError(
        "No results returned from thread execution, this should not happen.");
  } else {
    // Get service run options.
    TF_ASSIGN_OR_RETURN(auto stream,
                        backend().default_stream_executor()->CreateStream());
    ServiceExecutableRunOptions service_run_options =
        GetServiceRunOptionsForDevice(backend().default_device_ordinal(),
                                      stream.get(), &device_assignment, RunId(),
                                      backend().device_count());
    service_run_options.mutable_run_options()->set_execution_profile(profile);

    auto options = executable->module().config().debug_options();
    auto gpu_run_options = std::make_unique<gpu::GpuExecutableRunOptions>();
    if (options.xla_gpu_require_exclusive_lock()) {
      gpu_run_options->set_requires_exclusive_lock_on_gpu();
    }

    service_run_options.mutable_run_options()->set_gpu_executable_run_options(
        gpu_run_options.get());

    TF_ASSIGN_OR_RETURN(ExecutionOutput retval,
                        executable->ExecuteOnStreamWrapper(&service_run_options,
                                                          std::move(arguments[0])));
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
    return std::move(retval);
  }
}

absl::StatusOr<std::vector<Literal>> HloRunner::ExecuteReplicated(
    std::unique_ptr<HloModule> module, const ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<OpaqueExecutable> executable,
      CreateExecutable(std::move(module), options.run_hlo_passes));
  return ExecuteReplicated(executable.get(), options, device_assignment);
}

absl::StatusOr<std::vector<Literal>> HloRunner::ExecuteReplicatedImpl(
    std::function<absl::StatusOr<std::vector<std::vector<ScopedShapedBuffer>>>(
        const std::vector<ServiceExecutableRunOptions>&,
        const std::vector<absl::Span<const ShapedBuffer* const>>&)>
        execution_helper,
    std::function<int64_t(int64_t)> argument_count_provider,
    std::function<const Literal*(int64_t, int64_t)> argument_provider,
    const ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment) {
  return absl::UnimplementedError(
      "HloRunner::ExecuteReplicatedImpl is not implemented. "
      "Please use the HloRunner::ExecuteReplicated method instead.");
}

absl::StatusOr<std::vector<Literal>> HloRunner::ExecuteReplicated(
    OpaqueExecutable* executable, const ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment, ExecutionProfile* profile) {
  return absl::UnimplementedError(
      "HloRunner::ExecuteReplicatedImpl is not implemented. "
      "Please use the HloRunner::ExecuteReplicated method instead.");
}

absl::StatusOr<std::vector<Literal>> HloRunner::ExecuteReplicated(
    std::function<OpaqueExecutable*(int64_t)> executable_provider,
    std::function<int64_t(int64_t)> argument_count_provider,
    std::function<const Literal*(int64_t, int64_t)> argument_provider,
    const ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment) {
  return absl::UnimplementedError(
      "HloRunner::ExecuteReplicatedImpl is not implemented. "
      "Please use the HloRunner::ExecuteReplicated method instead.");
}

absl::StatusOr<std::vector<Literal>> HloRunner::ExecuteReplicated(
    std::unique_ptr<HloModule> module,
    const ReplicatedExecuteOptions& options) {
  TF_ASSIGN_OR_RETURN(
      DeviceAssignment device_assignment,
      backend().computation_placer()->AssignDevices(options.num_replicas, 1));
  return ExecuteReplicated(std::move(module), options, &device_assignment);
}

absl::StatusOr<std::unique_ptr<OpaqueExecutable>> HloRunner::CreateExecutable(
    std::unique_ptr<HloModule> module, bool run_hlo_passes) {
  return CreateExecutableWithBufferAssignment(
      std::move(module),
      /*buffer_assignment_proto=*/nullptr, run_hlo_passes);
}

absl::StatusOr<std::unique_ptr<OpaqueExecutable>>
HloRunner::CreateExecutableWithBufferAssignment(
    std::unique_ptr<HloModule> module,
    const BufferAssignmentProto* buffer_assignment_proto, bool run_hlo_passes) {
  MaybeUpdateEntryComputationLayout(module.get());
  if (run_hlo_passes) {
    if (buffer_assignment_proto != nullptr) {
      LOG(WARNING) << "Ignoring buffer assignment provided because hlo passes "
                      "are enabled.";
    }
    // Setup intra-op threads in module config
    if (backend().eigen_intra_op_thread_pool() != nullptr) {
      module->mutable_config().set_intra_op_parallelism_threads(
          backend().eigen_intra_op_thread_pool()->NumThreads());
    }
    auto module_group = std::make_unique<HloModuleGroup>(std::move(module));
    TF_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<Executable>> executables,
        backend().compiler()->Compile(std::move(module_group),
                                      {{backend().default_stream_executor()}},
                                      backend().memory_allocator()));
    return std::make_unique<HloRunnerExecutable>(this,
                                                 std::move(executables[0]));
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      backend().compiler()->RunBackendWithBufferAssignment(
          std::move(module), buffer_assignment_proto,
          backend().default_stream_executor(), backend().memory_allocator()));
  return std::make_unique<HloRunnerExecutable>(this, std::move(executable));
}

absl::StatusOr<std::unique_ptr<OpaqueExecutable>>
HloRunner::DeserializeExecutable(const absl::string_view serialized) const {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                      backend().compiler()->DeserializeExecutable(serialized));
  return std::make_unique<HloRunnerExecutable>(this, std::move(executable));
}

ServiceExecutableRunOptions HloRunner::GetServiceRunOptionsForDevice(
    int64_t device, se::Stream* stream, DeviceAssignment* device_assignment,
    RunId run_id, int local_device_count) {
  VLOG(2) << "Creating ServiceExecutableRunOptions for device " << device;
  ExecutableRunOptions run_options;
  run_options.set_device_ordinal(device);
  run_options.set_local_device_count(local_device_count);

  run_options.set_stream(stream);
  run_options.set_allocator(backend().memory_allocator());
  run_options.set_intra_op_thread_pool(
      backend().eigen_intra_op_thread_pool_device());
  if (device_assignment != nullptr) {
    run_options.set_device_assignment(device_assignment);
  }
  run_options.set_run_id(run_id);
  return ServiceExecutableRunOptions(run_options,
                                     backend().StreamBorrowerWithPriority());
}

Backend& HloRunner::backend() { return *backend_; }

const Backend& HloRunner::backend() const {
  return const_cast<HloRunner*>(this)->backend();
}

absl::string_view HloRunner::Name() const {
  return backend_->platform()->Name();
}

bool HloRunner::HasProperty(const HloRunnerPropertyTag::Type tag) const {
  if (tag == HloRunnerPropertyTag::kUsingGpuRocm) {
    const stream_executor::DeviceDescription& device_description =
        backend().default_stream_executor()->GetDeviceDescription();
    return std::holds_alternative<stream_executor::RocmComputeCapability>(
        device_description.gpu_compute_capability());
  }
  if (tag == HloRunnerPropertyTag::kCpu) {
    return backend().platform()->Name() == "Host";
  }
  return false;
}

absl::StatusOr<Executable*> HloRunner::ExecutableFromWrapped(
    const OpaqueExecutable* wrapped) const {
  TF_ASSIGN_OR_RETURN(const HloRunnerExecutable* const hlo_runner_executable,
                      HloRunnerExecutable::TryUnwrap(*this, wrapped));
  return hlo_runner_executable->executable();
}

absl::StatusOr<std::unique_ptr<Executable>> HloRunner::ExecutableFromWrapped(
    std::unique_ptr<OpaqueExecutable> wrapped) const {
  TF_ASSIGN_OR_RETURN(HloRunnerExecutable* const hlo_runner_executable,
                      HloRunnerExecutable::TryUnwrap(*this, wrapped.get()));
  return hlo_runner_executable->MoveExecutable();
}

std::unique_ptr<OpaqueExecutable> HloRunner::WrapExecutable(
    std::unique_ptr<Executable> executable) const {
  return std::make_unique<HloRunnerExecutable>(this, std::move(executable));
}

absl::StatusOr<const HloModule* absl_nonnull> HloRunner::HloModuleFromWrapped(
    const OpaqueExecutable* wrapped) const {
  TF_ASSIGN_OR_RETURN(const HloRunnerExecutable* const hlo_runner_executable,
                      HloRunnerExecutable::TryUnwrap(*this, wrapped));
  if (!hlo_runner_executable->executable()->has_module()) {
    return absl::NotFoundError("Executable has no module.");
  }
  return &hlo_runner_executable->executable()->module();
}

absl::StatusOr<const HloProto* absl_nonnull> HloRunner::HloProtoFromWrapped(
    const OpaqueExecutable* wrapped) const {
  TF_ASSIGN_OR_RETURN(const HloRunnerExecutable* const hlo_runner_executable,
                      HloRunnerExecutable::TryUnwrap(*this, wrapped));
  return hlo_runner_executable->executable()->hlo_proto();
}

void HloRunner::MaybeUpdateEntryComputationLayout(HloModule* module) {
  absl::MutexLock lock(&mu_);
  if (module_ids_with_updated_layouts_.insert(module->unique_id()).second) {
    VLOG(2) << "Updating entry computation layout";
    xla::UpdateEntryComputationLayout(module, device_shape_representation_fn_);
  }
}
}  // namespace xla
