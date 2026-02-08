# (Load model and weights first)
model.eval()
seed_input = torch.tensor([[8]]).to(device) # Replace 8 with your ID for 'በ'

with torch.no_grad():
    prediction = model(seed_input)
    next_token_id = torch.argmax(prediction[0, -1]).item()
    print(f"The AI thinks the next splinter after 'በ' is ID: {next_token_id}")