import os
import torch
from datetime import datetime
from tqdm.auto import tqdm


def acc(pred, real):
    return (pred.argmax(axis=1) == real).cpu().numpy().sum() / len(pred)


def eval_model(model, val_dataloader, loss_fn, device):
    with torch.no_grad():
        val_loss_sum, val_acc_sum, step = 0., 0., 0
        for batch_x1, *batch_x2, batch_y in val_dataloader:
            model.eval()
            batch_x1 = batch_x1.to(device)
            batch_x2 = [i.to(device) for i in batch_x2]
            batch_y = batch_y.to(device)
            pred = model(batch_x1, *batch_x2)
            val_loss_sum += loss_fn(pred, batch_y)
            val_acc_sum += acc(pred, batch_y)
            step += 1
        val_loss = val_loss_sum / step
        val_acc = val_acc_sum / step
        picture_val_acc = eval_model_mode(model, val_dataloader, device, mode='picture')
        text_val_acc = eval_model_mode(model, val_dataloader, device, mode='text')

        return val_loss, val_acc, picture_val_acc, text_val_acc


def eval_model_mode(model, val_dataloader, device, mode):
    with torch.no_grad():
        val_acc_sum, step = 0., 0
        for batch_x1, *batch_x2, batch_y in val_dataloader:
            model.eval()
            batch_x1 = batch_x1.to(device)
            batch_x2 = [i.to(device) for i in batch_x2]
            batch_y = batch_y.to(device)
            pred = model(batch_x1, *batch_x2, mode, device)
            val_acc_sum += acc(pred, batch_y)
            step += 1
        val_acc = val_acc_sum / step
        return val_acc


def train_model(model, train_dataloader, val_dataloader, loss_fn, optimizer, num_epochs, device, save):
    best_acc = 0
    log_dir = f'./logs/{model.fusion_level}_fusion'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'metrics.csv')
    f = open(log_file, 'w')
    f.write('epoch,train_loss,train_acc,val_loss,val_acc,picture_val_acc,text_val_acc\n')
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, step = 0., 0., 0
        loop = tqdm(train_dataloader, total=len(train_dataloader), leave=False)
        for batch_x1, *batch_x2, batch_y in loop:
            model.train()
            batch_x1 = batch_x1.to(device)
            batch_x2 = [i.to(device) for i in batch_x2]
            batch_y = batch_y.to(device)
            pred = model(batch_x1, *batch_x2)
            loss = loss_fn(pred, batch_y)
            step_acc = acc(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss
            train_acc_sum += step_acc
            step += 1
            loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
            loop.set_postfix(loss=loss.item(), acc=step_acc)

        train_loss = train_loss_sum / step
        train_acc = train_acc_sum / step

        val_loss, val_acc, picture_val_acc, text_val_acc = eval_model(model, val_dataloader, loss_fn, device)
        print('Epoch:', '%03d' % epoch,
              '|train loss =', '%06f' % train_loss,
              '|train acc =', '%06f' % train_acc,
              '|val loss =', '%06f' % val_loss,
              '|val acc =', '%06f' % val_acc,
              '|picture val acc =', '%06f' % picture_val_acc,
              '|text val acc =', '%06f' % text_val_acc,
              end=' ')
        f.write(f'{epoch},{train_loss},{train_acc},{val_loss},{val_acc},{picture_val_acc},{text_val_acc}\n')

        if save and best_acc <= val_acc:
            save_dir = f'./trained_model/{model.fusion_level}_fusion'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
            best_acc = val_acc
            print('|model saved')
        else:
            print()
    f.close()
