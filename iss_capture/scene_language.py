"""Use the same frozen pretrained token embeddings as SmolVLA for task semantics."""
import torch

def language_embedding(base, batch):
    tokens=batch['observation.language.tokens']
    mask=batch['observation.language.attention_mask']
    with torch.no_grad():
        embedded=base.model.vlm_with_expert.embed_language_tokens(tokens).float()
        return (embedded*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)

# Training augmentation, not a runtime keyword parser or instruction dispatcher.
PARAPHRASES={
    'task1': [
        'Use the right-hand sponge to collect the floating water droplet. Leave the left arm still.',
        'Catch the rising droplet using the sponge held in the right gripper.',
        'Move the right arm so the water touches the exposed sponge surface.',
        'Collect the water droplet with the right sponge and hold it.',
    ],
    'task2': [
        'Grasp the floating orange pen using the left hand. Keep the right arm still.',
        'Catch and securely hold the rotating pen with the left gripper.',
        'Use the left arm to take the orange pen between the arms.',
        'Grab the pen with the left hand and keep holding it.',
    ],
    'task3': [
        'Hold the pen in the left hand and the cap in the right hand, then put the cap onto the pen.',
        'Use both arms to grasp the orange pen and blue cap and assemble them.',
        'Left hand catches the pen, right hand catches the cap. Align and insert the pen tip into the cap.',
        'Cap the pen using the left gripper for the pen and right gripper for the cap.',
    ],
}
