import torch
import torch.autograd as torch_ad

from firedrake.assemble import assemble
from neural_network_operators import NeuralNet


class CustomOperator(torch_ad.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    # This method is wrapped by something cancelling annotation (probably 'with torch.no_grad()')
    def forward(ctx, metadata, *θ):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        N = metadata['N']
        ctx.metadata.update(metadata)
        with torch.enable_grad():
            assembled_N = assemble(N)
            # Find out how to do that cleanly knowing that you can't use
            # single dispatch (since N can only be inside metadata for backward)
            # and you need to have at least hit `FiredrakeLoss.__call__` to do the dispatch.
            if isinstance(N, NeuralNet):
                output = N.model_output.squeeze(0)
            else:
                print('\n\n Form!!!')
                output = convert_to_torch(assembled_N)
        ctx.save_for_backward(output)  # N.model_output.squeeze(0))
        return output.detach()  # convert_to_torch(assembled_N)

# Dispatch strategy for forward depending on whether ExternalOperator or not. Concretely what changes is:    
#    forward: form the tensor from assembled result instead of model_output
#    Backward: call assemble on action(adjoint(derivative(...)), ...) and parameters should be updated
#              then return their grad as you do it now

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        output, = ctx.saved_tensors
        N = ctx.metadata['N']
        with torch.enable_grad():
            # uu, = N.operator_inputs()
            # xx = convert_to_torch(uu)
            # output = N.model(xx.unsqueeze(0)).squeeze(0)
            if isinstance(N, NeuralNet):
                output.backward(grad_output)
            else:
                import ipdb; ipdb.set_trace()
        return None, *[θ.grad for θ in N.model.parameters()]
        """
        # delta_N = convert_to_firedrake(y.grad)
        delta_N = convert_to_firedrake(dy, V)
        m, *_ = N.operator_params()
        dNdm = action(adjoint(expand_derivatives(derivative(N, m))), delta_N)
        with torch.enable_grad():
            w = assemble(dNdm)
        import ipdb; ipdb.set_trace()
        w = convert_to_torch(w)
        """
