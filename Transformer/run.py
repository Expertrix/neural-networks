import torch
from torch import nn
from transformer import Transformer

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X = torch.tensor([
        [1, 3, 5, 4, 6, 2, 0, 0],
        [1, 4, 3, 6, 7, 9, 8, 2]
    ]).to(device)
    y = torch.tensor([
        [1, 6, 4, 5, 3, 2, 0, 0],
        [1, 8, 9, 7, 6, 3, 4, 2]
    ]).to(device)
    pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10

    model = Transformer(src_vocab_size, trg_vocab_size,
                        pad_idx, pad_idx, device=device).to(device)

    # =================================================================

    print('\n=== Training ===')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    epochs = 10
    for i in range(epochs):
        print(f'--- Epoch {i+1} / {epochs} ---')

        out = model(X, y[:, :-1])
        out = out.reshape(-1, out.shape[2])

        target = y[:, 1:].reshape(-1)  

        loss = criterion(out, target)
        print(f"Loss {loss.item()}")
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

    # =================================================================

    print('\n=== Evaluation ===')
    out = model(X, y[:, :-1])

    print("Input:", X)
    print("Prediction:", out.argmax(2))
    print("Ground Truth:", y)

    # =================================================================

    print("\n=== Auto Regressive Inference ===")
    preds = [1]  
    max_gen_len = 8

    for i in range(max_gen_len):

        # print(f'--- Token {i+1} ---')

        input_tensor = torch.tensor(preds).unsqueeze(0).to(device)
        # print('Input Tensor:', input_tensor)

        with torch.no_grad():
            out = model(X[:1, :], input_tensor)
            # print('Raw Output Shape:', out.shape)

        word_idx = out.argmax(dim=-1)[:, -1].item()
        # print('Next Word Index:', word_idx)


        preds.append(word_idx)

        if word_idx == 2: 
            break

    print("Prediction:", preds)
    print("Ground Truth:", y[:1, :])