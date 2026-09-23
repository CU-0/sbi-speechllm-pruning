import torch

def remove_layers(layers_module, indicies_to_remove):
    """
    layers_module: nn.ModuleList of layers (encoder or decoder)
    indicies_to_remove: list of indices of layers to remove
    Returns: new nn.ModuleList with the specified layers removed.
    """
    # Get indices of the least important layers
    
    # Create a new ModuleList excluding the least important layers
    new_layers = torch.nn.ModuleList([layer for i, layer in enumerate(layers_module) if i not in indicies_to_remove])

    # rewire the layer_idx attributes of the remaining layers to reflect their new positions if necessary
    for new_idx, layer in enumerate(new_layers):
        attn_module = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn_module is not None and hasattr(attn_module, "layer_idx"):
            attn_module.layer_idx = new_idx
    
    return new_layers

if __name__ == "__main__":
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    # test the pruning function with a small example
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")

    remove_n = 2  # Number of layers to remove for demonstration
    model.model.audio_tower.layers, removed_encoder_indices = remove_layers(model.model.audio_tower.layers, [1,2,3])
    model.model.language_model.layers, removed_decoder_indices = remove_layers(model.model.language_model.layers, [1,2,3])

    print("Removed encoder layer indices:", removed_encoder_indices)
    print("Removed decoder layer indices:", removed_decoder_indices)

    # Update the model configuration to reflect the new number of layers
    model.config.text_config.num_hidden_layers = len(model.model.language_model.layers)
    model.config.audio_config.encoder_layers = len(model.model.audio_tower.layers)
    # Update the layer_types list to remove the corresponding entries for the removed decoder layers    
    model.config.text_config.layer_types = [
        lt for i, lt in enumerate(model.config.text_config.layer_types)
        if i not in removed_decoder_indices
    ]

    # save the modified model to a new directory
    model.save_pretrained(f"Qwen/Qwen2-Audio-7B-Instruct-pruned{remove_n}")
    processor.save_pretrained(f"Qwen/Qwen2-Audio-7B-Instruct-pruned{remove_n}")
    print("Pruned model saved to Qwen/Qwen2-Audio-7B-Instruct-pruned.")

        # Load the pruned model to verify it works
    pruned_model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct-pruned", device_map="auto")
    pruned_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct-pruned")
    print("Pruned model loaded successfully.")