from transformers import Trainer

class CustomMultimodalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get additional diffusion parameters from inputs
        latent = inputs.pop("latent")
        timesteps = inputs.pop("timesteps")
        
        # Forward pass returns a tuple (mllm_output, diffusion_output, decoded_images)
        mllm_output, diffusion_output, decoded_images = model(**inputs, latent=latent, timesteps=timesteps)
        
        # Compute losses (assuming you have labels in inputs)
        lm_loss = None
        if "labels" in inputs:
            lm_loss = nn.CrossEntropyLoss()(
                mllm_output.logits.view(-1, mllm_output.logits.size(-1)),
                inputs["labels"].view(-1)
            )
        mse_loss = nn.MSELoss()(diffusion_output.sample, inputs["noise"])  # Ensure 'noise' is part of your inputs
        
        # Combine the losses; adjust lambda as necessary
        total_loss = mse_loss + (lm_loss if lm_loss is not None else 0)
        
        return (total_loss, (mllm_output, diffusion_output, decoded_images)) if return_outputs else total_loss

# Then, initialize your Trainer:
trainer = CustomMultimodalTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # additional parameters as needed
)
trainer.train()
