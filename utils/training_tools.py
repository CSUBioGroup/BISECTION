import time

import torch

from .utils import AverageMeter, ProgressMeter, accuracy

def train_distiller(train_loader, student, teacher, criterion, optimizer, device):
    student = student.to(device)
    teacher = teacher.to(device)
    loss_epoch = 0.0
    count = 0

    for i, (x, pseu, _) in enumerate(train_loader):
        x = x.to(device)
        pseu = pseu.to(device)
        
        count += 1
        optimizer.zero_grad()
        
        with torch.no_grad():
            teacher_logits = teacher(x)
            
        if student.attention_flag:
            student_logits, alphas, gene_scores = student(x)
        else:
            student_logits = student(x)
                    
        loss = criterion(student_logits, teacher_logits, pseu)
        
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print(f"Step [{i}]\t loss_instance: {loss.item()}")
        
        loss_epoch += loss.item()

    loss_epoch = loss_epoch / count
        
    return loss_epoch

def train_distiller_with_gene_set(train_loader, 
                                  student, 
                                  teacher, 
                                  criterion, 
                                  optimizer, 
                                  device):
    student = student.to(device)
    teacher = teacher.to(device)
    loss_epoch = 0.0
    count = 0

    for i, (x, pseu, _) in enumerate(train_loader):
        x = x.to(device)
        pseu = pseu.to(device)
        
        count += 1
        optimizer.zero_grad()
        
        with torch.no_grad():
            teacher_logits = teacher(x)
        
       
        student_logits = student(x)
        
        loss = criterion(student_logits, teacher_logits, pseu)
        
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print(f"Step [{i}]\t loss_instance: {loss.item()}")
        
        loss_epoch += loss.item()

    loss_epoch = loss_epoch / count
        
    return loss_epoch

def train_teacher(train_loader, teacher, criterion, optimizer, device, epoch):

    teacher = teacher.to(device)
    loss_epoch = 0.0
    count = 0
    
    teacher.eval()
    end = time.time()
    for i, (x, pseu, y) in enumerate(train_loader):

        x = x.to(device)
        pseu = pseu.to(device)

        count += 1

        out = teacher(x)
        loss = criterion(out, pseu)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Step [{i}]\t loss_instance: {loss.item()}")

        loss_epoch += loss.item()

    loss_epoch = loss_epoch / count

    return loss_epoch
    
def validate_teacher(val_loader, model, criterion, device):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    progress = ProgressMeter(len(val_loader), 
                             [batch_time, losses, top1], 
                             prefix="Test: ")
    
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (x, pseu, y) in enumerate(val_loader):
            x = x.to(device)
            pseu = pseu.to(device)
            
            output = model(x)
            loss = criterion(output, pseu)
            acc1 = accuracy(output, pseu, topk=(1, ))
            
            losses.update(loss.item(), x.size(0))
            top1.update(acc1[0].item(), x.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                progress.display(i)

        print(f"* Acc@1 {top1.avg:.3f}")

    return top1.avg

def get_prediction(model, device, val_loader, name="teacher"):
    model.eval()
    pred = []
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)

            if name == "teacher":
                output = model(x)
            if name == "student":
                output = model(x)

            _, res = torch.max(output, dim=1)
            res = res.detach().cpu().numpy()
            
            pred.extend(res)

    if name == "teacher":
        return pred
    
    return pred

def get_prediction_with_gene_set(model, device, val_loader, name="teacher"):
    model.eval()
    pred = []
    with torch.no_grad():
        for _, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)

            if name == "teacher":
                output = model(x)
            if name == "student" :
                output = model(x)

            _, res = torch.max(output, dim=1)
            res = res.detach().cpu().numpy()
            
            pred.extend(res)
    
    return pred


