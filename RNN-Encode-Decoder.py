#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import Dataset, DataLoader
import os, os.path 
import numpy as np
import pickle
from glob import glob
from tqdm import tqdm, trange

"""Change to the data folder"""
new_path = "./new_train/new_train"

cuda_status = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# number of sequences in each dataset
# train:205942  val:3200 test: 36272 
# sequences sampled at 10HZ rate


# ### Create a dataset class 

# In[ ]:


class ArgoverseDataset(Dataset):
    """Dataset class for Argoverse"""
    def __init__(self, data_path: str, transform=None):
        super(ArgoverseDataset, self).__init__()
        self.data_path = data_path
        self.transform = transform

        self.pkl_list = glob(os.path.join(self.data_path, '*'))
        self.pkl_list.sort()
        
    def __len__(self):
        return len(self.pkl_list)

    def __getitem__(self, idx):

        pkl_path = self.pkl_list[idx]
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        if self.transform:
            data = self.transform(data)

        return data


# intialize a dataset
val_dataset  = ArgoverseDataset(data_path=new_path)


# ### Create a loader to enable batch processing

# In[ ]:


batch_sz = 512

def my_collate(batch):
    """ collate lists of samples into batches, create [ batch_sz x agent_sz x seq_len x feature] """
    inp = [np.dstack([scene['p_in'], scene['v_in']]) for scene in batch]
    out = [np.dstack([scene['p_out'], scene['v_out']]) for scene in batch]
    scene_ids = [scene['scene_idx'] for scene in batch]
    track_ids = [scene['track_id'] for scene in batch]
    agent_ids = [scene['agent_id'] for scene in batch]
    car_mask = [scene['car_mask'] for scene in batch]
    inp = torch.LongTensor(inp)
    out = torch.LongTensor(out)
    scene_ids = torch.LongTensor(scene_ids)
    car_mask = torch.LongTensor(car_mask)

    num_cars = np.zeros((inp.shape[0]))
    offsets = np.zeros((inp.shape[0], 2))
    
    for i in range(inp.shape[0]):
        num_vehicles = 0
        for j in range(60):
            if car_mask[i][j][0] == 1:
                num_vehicles += 1
        num_cars[i] = num_vehicles
        
        agent_id = agent_ids[i]
        vehicle_index = 0
        found = False
        while not found:
            if track_ids[i][vehicle_index][0][0] == agent_id:
                found = True
            else:
                vehicle_index += 1
        start_x = inp[i][vehicle_index][0][0]
        start_y = inp[i][vehicle_index][0][1]
        
        offsets[i][0] = start_x
        offsets[i][1] = start_y
        
        inp[i,0:num_vehicles,:,0] -= start_x
        inp[i,0:num_vehicles,:,1] -= start_y
        #out[i,0:num_vehicles,:,0] -= start_x
        #out[i,0:num_vehicles,:,1] -= start_y
        
    offsets = torch.LongTensor(offsets)
    num_cars = torch.LongTensor(num_cars)

    return [inp, out, scene_ids, track_ids, agent_ids, car_mask, num_cars, offsets]

def test_collate(batch):
    """ collate lists of samples into batches, create [ batch_sz x agent_sz x seq_len x feature] """
    inp = [np.dstack([scene['p_in'], scene['v_in']]) for scene in batch]
    scene_ids = [scene['scene_idx'] for scene in batch]
    track_ids = [scene['track_id'] for scene in batch]
    agent_ids = [scene['agent_id'] for scene in batch]
    car_mask = [scene['car_mask'] for scene in batch]
    
    inp = torch.FloatTensor(inp)
    scene_ids = torch.LongTensor(scene_ids)
    car_mask = torch.LongTensor(car_mask)
    
    num_cars = np.zeros((inp.shape[0]))
    offsets = np.zeros((inp.shape[0], 2))
    
    for i in range(inp.shape[0]):
        num_vehicles = 0
        for j in range(60):
            if car_mask[i][j][0] == 1:
                num_vehicles += 1
        num_cars[i] = num_vehicles
        
        agent_id = agent_ids[i]
        vehicle_index = 0
        found = False
        while not found:
            if track_ids[i][vehicle_index][0][0] == agent_id:
                found = True
            else:
                vehicle_index += 1
        start_x = inp[i][vehicle_index][0][0]
        start_y = inp[i][vehicle_index][0][1]
        
        offsets[i][0] = start_x
        offsets[i][1] = start_y
        
        inp[i,0:num_vehicles,:,0] -= start_x
        inp[i,0:num_vehicles,:,1] -= start_y
        
    offsets = torch.LongTensor(offsets)
    num_cars = torch.LongTensor(num_cars)
    return [inp, scene_ids, track_ids, agent_ids, car_mask, num_cars, offsets]

#temp_loader = DataLoader(val_dataset,batch_size=205942, shuffle = False, collate_fn=my_collate, num_workers=0, drop_last=True)
#data = next(iter(temp_loader))
#inp, out, scene_ids, track_ids, agent_ids, car_mask = data
#print(inp.mean())
#print(inp.std())
#print(out.mean())
#print(out.std())

val_loader = DataLoader(val_dataset,batch_size=batch_sz, shuffle = False, collate_fn=my_collate, num_workers=0, drop_last=True)


# In[ ]:


class RNNEncoderDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.RNN(240, hidden_size=512, batch_first=True, nonlinearity='relu')
        self.decoder = torch.nn.RNN(240, hidden_size=512, batch_first=True, nonlinearity='relu')
        
        self.align1 = torch.nn.Linear(10240, 19)
        #self.attn = Attention(512,512)
        
        self.linear = torch.nn.Linear(512, 240)

    def forward(self, x, y, teach = False, teaching_ratio = 0.5):
        batch_size = x.shape[0]
        
        # batch_szx60x19x4
        x = x.permute(0,2,1,3)
        x = torch.flatten(x, start_dim=2)
        # batch_szx19x240
        output, hidden = self.encoder(x)
        
        if cuda_status:
            outputs = torch.zeros(30,batch_size,60,4).to(device).cuda()
            dec_out, dec_hidden = self.decoder(torch.full((batch_size,1,240), -1).to(device).cuda(), hidden)
        else:
            outputs = torch.zeros(30,batch_size,60,4).to(device)
            dec_out, dec_hidden = self.decoder(torch.full((batch_size,1,240), -1).to(device).float(), hidden)
        
        # dec_out: batch_szx1x512
        dec_out = dec_out.permute(1,0,2).squeeze(0)
        # batch_szx512
        dec_out = self.linear(dec_out)
        # batch_sz x 240
        outputs[0] = torch.reshape(dec_out, torch.Size([batch_size, 60, 4]))
        
        if teach:
            next_in = torch.flatten(y[:,:,0,:].squeeze(2), start_dim=1).unsqueeze(1)
            # batch_szx240
        else:
            next_in = dec_out.unsqueeze(1)
        
        # output: batch_szx19x512
        # h_n: 1xbatch_szx512
        prevState = hidden.permute(1,0,2)
        inputStates = output
        
        for i in range(1,30):
            alignment = self.align1(torch.flatten(torch.cat((inputStates, prevState), 1), start_dim=1))
            #batch_szx19
            
            attention = torch.nn.functional.softmax(alignment, dim=1)
            attention = attention.unsqueeze(1)
            #batch_szx1x19
            
            new_hidden = torch.bmm(attention, inputStates)
            new_hidden = new_hidden.permute(1,0,2)
            #1xbatch_szx512
             
            dec_out, dec_hidden = self.decoder(next_in, new_hidden)
            # dec_out: batch_szx1x512
            dec_out = dec_out.permute(1,0,2).squeeze(0)
            # batch_szx512
            dec_out = self.linear(dec_out)
            # batch_sz x 240
            
            teaching = random.random() < teaching_ratio
            
            if teach and teaching:
                next_in = torch.flatten(y[:,:,i-1,:].squeeze(2), start_dim=1).unsqueeze(1)
            else:
                next_in = dec_out.unsqueeze(1)
                
            outputs[i] = torch.reshape(dec_out, torch.Size([batch_size, 60, 4]))
            
            prevState = dec_hidden.permute(1,0,2)
        
        return outputs.permute(1,2,0,3)

model = RNNEncoderDecoder()
model.to(device)
if cuda_status:
    model = model.cuda()


# ### Visualize the batch of sequences

# In[ ]:


import matplotlib.pyplot as plt
import random
from tqdm.notebook import tqdm

agent_id = 0
epoch = 6
        
# Use the nn package to define our loss function
loss_fn=torch.nn.MSELoss()

# Use the optim package to define an Optimizer

learning_rate =1e-3
#learning_rate =0.01
#optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i in trange(epoch):
    iterator = tqdm(val_loader)
    total = 0
    count = 0
    
    for i_batch, sample_batch in enumerate(iterator):
        inp, out, scene_ids, track_ids, agent_ids, car_mask, num_cars, offsets = sample_batch
        
        x = inp.float()
        y = out.float()

        if cuda_status:
            x.to(device)
            y.to(device)
            x = x.cuda()
            y = y.cuda()

        y_pred = None
        # Forward pass: predict y by passing x to the model.    
        y_pred = model(x, y, False)
        #y_pred = torch.reshape(y_pred, torch.Size([batch_sz, 60, 30, 4]))

        for i in range(y_pred.shape[0]):
            y_pred[i,0:num_cars[i],:,0] += offsets[i][0]
            y_pred[i,0:num_cars[i],:,1] += offsets[i][1]

        # Compute the loss.
        loss = loss_fn(y_pred, y)
        total += torch.sqrt(loss).item()

        # Before backward pass, zero outgradients to clear buffers  
        optimizer.zero_grad()

        # Backward pass: compute gradient w.r.t modelparameters
        loss.backward()

        # makes an gradient descent step to update its parameters
        optimizer.step()

        count += 1
        
        iterator.set_postfix(loss=total / count, curr=torch.sqrt(loss).item())


# In[ ]:


torch.save(model, './models/6epoch-RNN-Encoder-Decoder-Attention-512-reoriented.pt')


# In[ ]:





# In[ ]:


import pandas as pd

# Submission output
writeCSV = True
val_path = "./new_val_in/new_val_in"

if writeCSV:
    
    dataset = ArgoverseDataset(data_path=val_path)
    test_loader = DataLoader(dataset,batch_size=64, shuffle = False, collate_fn=test_collate, num_workers=0)
    
    data = []
    
    with torch.no_grad():
        for i_batch, sample_batch in enumerate(tqdm(test_loader)):
            inp, scene_ids, track_ids, agent_ids, car_mask, num_cars, offsets = sample_batch

            if cuda_status:
                model = model.cuda()
                x = inp.float().cuda()
            else:
                x = inp.float()

            y_pred = None

            # Forward pass: predict y by passing x to the model.    
            y_pred = model(x, None, False)
            #y_pred = torch.reshape(y_pred, torch.Size([64, 60, 30, 4]))
            
            for i in range(y_pred.shape[0]):
                y_pred[i,0:num_cars[i],:,0] += offsets[i][0]
                y_pred[i,0:num_cars[i],:,1] += offsets[i][1]

            
            for i in range(64):
                row = []
                row.append(scene_ids[i].item())
                curr = y_pred[i]
                
                agent_id = agent_ids[i]
                
                for j in range(30):
                    vehicle_index = 0
                    found = False
                    while not found:
                        if track_ids[i][vehicle_index][j][0] == agent_id:
                            found = True
                        else:
                            vehicle_index += 1

                    row.append(str(curr[vehicle_index][j][0].item()))
                    row.append(str(curr[vehicle_index][j][1].item()))
                    
                data.append(row)

    df = pd.DataFrame(data, columns = ['ID','v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','v11','v12','v13','v14','v15','v16','v17','v18','v19','v20','v21','v22','v23','v24','v25','v26','v27','v28','v29','v30','v31','v32','v33','v34','v35','v36','v37','v38','v39','v40','v41','v42','v43','v44','v45','v46','v47','v48','v49','v50','v51','v52','v53','v54','v55','v56','v57','v58','v59','v60'])
    print(df)
    df.to_csv('submission.csv', index=False)
                
                
                


