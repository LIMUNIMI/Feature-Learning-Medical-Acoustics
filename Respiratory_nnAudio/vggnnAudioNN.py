from nnAudio import features
import torch
import matplotlib.pyplot as plt
import torchaudio
import numpy as np
import pandas as pd
import torchvision
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
#import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score

torch.cuda.empty_cache()

##### GPU #####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if str(device) == "cuda":
    num_workers = 1
    pin_memory = True
    print("device: gpu")
else:
    num_workers = 0
    pin_memory = False
    print("device: cpu")




BATCH_SIZE = 64
EPOCHS = 200
LEARN_RATE = 1e-5





##################### DATASET #######################
def createDS(): 
    frames = 8000
    batch_size = BATCH_SIZE
    path = '../Respiratory_Dataset_SEGMENTED_ICBHI/'
    
    df = pd.read_csv(path+'Reference_respiro2label.csv')

    
    label = df['class_label']
    label = torch.LongTensor(np.array(label))
    file_path = df['name']

    


    def align(samples: torch.Tensor, seq_len: int = frames):
        pad_length = seq_len - samples.size(dim=1)
        return torch.nn.functional.pad(samples, [0, pad_length])


    ds = torch.zeros(0,1,frames)

    for x in range(len(file_path)):
        filename = path+file_path[x]
        audio_sample, sr = torchaudio.load(filename, num_frames = frames)
        audio_sample = align(audio_sample)
        audio_sample = audio_sample[None, :, :]
        ds = torch.cat((ds, audio_sample))

    print("- Dimensione ds--> ", ds.shape)

    
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10


    ds_train, ds_test, label_train, label_test = train_test_split(ds,label, test_size=1 - train_ratio, random_state=101, shuffle=True)
    

    ds_val, ds_test, label_val, label_test = train_test_split(ds_test, label_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=101, shuffle=True) 

    print("Dimensione train_ds  ---> ", ds_train.shape)
    print("Dimensione valitation_ds  ---> ", ds_val.shape)
    print("Dimensione test_ds  ---> ", ds_test.shape)
    
    
    train_set = torch.utils.data.TensorDataset(ds_train, label_train)
    val_set = torch.utils.data.TensorDataset(ds_val, label_val)
    test_set = torch.utils.data.TensorDataset(ds_test, label_test)
    
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    print("NNAUDIO DATASET DONE")
    return train_loader, test_loader, val_loader





##################### MODEL ######################
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.mel = features.MelSpectrogram(n_mels=128, hop_length=10, win_length= 30, n_fft=512, 
                              window='hann', center=True, pad_mode='reflect',
                              fmin=100, fmax=2000, sr=4000, trainable_mel=True, trainable_STFT=True, verbose=True,
                              htk=True)
        

        self.vgg16 = torchvision.models.vgg16(pretrained = False, num_classes = 2)


    def forward(self, x):
        x = self.mel(x) #(batch_size, freq_bins, time_steps)        
        
        x = torchvision.transforms.functional.resize(x, size=(224, 224))
        x = x.unsqueeze(1)
        x = torch.cat((x,)*3, axis=1) 
        
        return self.vgg16(x)









valid_loss = []
losses = []
trainloss = []
def train(model, epoch, log_interval, train_loader, val_loader, optimizer, loss_function):
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/nnAudio'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as profiler:

        trainlossep = 0.0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()


            #profiler.step()

            # Print training stats
            if batch_idx % log_interval == 0:
                print(f">>>>>Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
            # record loss
            losses.append(loss.item())
            trainlossep += loss.item()
        trainloss.append(trainlossep/len(train_loader))



    valid_loss_ep = 0.0
    
    model.eval()     
    for data, labels in val_loader:
        data = data.to(device)
        labels = labels.to(device)
        
        target = model(data)
        loss = loss_function(target,labels)
        valid_loss_ep +=loss.item()
    valid_loss.append(valid_loss_ep/len(val_loader))
    print('validation loss ===> ', valid_loss[-1])





def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def test(model, epoch, test_loader):
    model.eval()
    correct = 0.0
    predlist=torch.zeros(0,dtype=torch.long)
    targetlist=torch.zeros(0,dtype=torch.long)

    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)

        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # Append batch prediction results
        predlist=torch.cat([predlist,pred.view(-1).cpu()])
        targetlist=torch.cat([targetlist,target.view(-1).cpu()])

    print("Test Epoch: "+str(epoch)+"     Accuracy: "+str(correct)+"/"+str(len(test_loader.dataset))+"  "+str(correct / len(test_loader.dataset)))
    
    #Confusion Matrix
    cm = confusion_matrix(targetlist.numpy(), predlist.numpy())
    print(cm)
    #Accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(targetlist.numpy(), predlist.numpy())
    print('Accuracy: %f' % accuracy)
    #Precision tp / (tp + fp)
    precision = precision_score(targetlist.numpy(), predlist.numpy())
    print('Precision: %f' % precision)
    #Recall: tp / (tp + fn)
    recall = recall_score(targetlist.numpy(), predlist.numpy())
    print('Recall: %f' % recall)
    print("BALANCED Accuracy --> ", balanced_accuracy_score(targetlist.numpy(), predlist.numpy()))  

    f = open("resp_vgg_nnaudio.txt", "a+")
    f.write("***************** Predizione ******** Epoca: %d \n" % epoch)
    f.write(np.array2string(predlist.numpy()))
    f.write("\n")


    return cm


def build_model():
    model = Model()

    model = torch.nn.DataParallel(model)
    model.to(device)

    # Conto il numero di parametri del modello
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    n = count_parameters(model)
    print("Number of trainable parameters: %s" % n)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    loss_function = torch.nn.CrossEntropyLoss() 


    def conta(model):
        total_params=0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            print(name, param)
            total_params+=param
        print("total param ->->->->",total_params)


    conta(model)


    return model, optimizer, loss_function

########################## TRAINING AND TEST ######################

def train_test(modello, ottimizzatore, perdita):

    model = modello
    optimizer = ottimizzatore
    loss_function = perdita


    train_loader, test_loader, val_loader = createDS()
    log_interval = 500
    n_epoch = EPOCHS

    f = open("resp_vgg_nnaudio.txt", "w+")
    f.write("Respiratory VGG16 nnAudio \n")
    f.close()

    

    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval, train_loader, val_loader, optimizer, loss_function)
        cm = test(model, epoch, test_loader)
            
    print("----------> end <-----------")

    #plt.figure(figsize=(10,5))
    #plt.plot(trainloss, label="train")
    #plt.plot(valid_loss, label="val")
    #plt.xlabel("epoch")
    #plt.ylabel("loss")
    #plt.legend()
    #plt.savefig("vgglossNN.png")


    #estraggo spettrogramma post-training
    testaudio, srate = torchaudio.load('../resp_121_1b1_Tc_sc_Meditron_1.wav', num_frames = 8000)

    post = model.mel(testaudio.cuda())
    post = torch.squeeze(post)
    plt.imshow(post.cpu().detach().numpy(), origin='lower', aspect='auto')
    plt.title('Trained nnAudio', size=18)
    plt.savefig("trained_nnAudio_resp.png")



