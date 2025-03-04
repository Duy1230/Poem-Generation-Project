import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer


@st.cache_resource
def load_model():
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2_viet_poem_generation")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2_viet_poem_generation")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, tokenizer, device


model, tokenizer, device = load_model()

# Interface layout
st.title("🎭 Vietnamese Poem Generator")
st.markdown("Generate poetic verses with AI")

# Input controls
with st.sidebar:
    st.header("Generation Parameters")
    max_length = st.number_input("Max length", 50, 500, 150, step=10)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8)
    top_k = st.slider("Top-k", 1, 100, 50)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95)
    repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.2)

# Main input
prompt = st.text_area("Enter your starting lines:",
                      "Học học nữa học mãi\n", height=100)

# Generation button
if st.button("Generate Poem"):
    with st.spinner("Composing poetic masterpiece..."):
        try:
            # Tokenize input
            inputs = tokenizer(
                prompt, return_tensors="pt").input_ids.to(device)

            # Generate text
            outputs = model.generate(
                inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id
            )

            # Decode and display
            poem = tokenizer.decode(outputs[0], skip_special_tokens=True)

            st.divider()
            st.subheader("Generated Poem")
            st.markdown(f"```\n{poem}\n```")

        except Exception as e:
            st.error(f"Error generating poem: {str(e)}")

# Add some instructions
with st.expander("💡 Usage Tips"):
    st.markdown("""
    - Start with a meaningful phrase or line (2-5 words works best)
    - Adjust temperature for creativity (lower=more focused, higher=more random)
    - Use Top-k/p to control word selection diversity
    - Higher repetition penalty reduces repeated phrases
    """)

# Optional: Add system info
if device == "cuda":
    st.sidebar.success("✅ Using GPU acceleration")
else:
    st.sidebar.warning("⚠️ Using CPU - generation will be slower")

st.sidebar.markdown("---")
st.sidebar.caption("Poem Generation System v1.0 | GPT-2 Fine-tuned Model")
