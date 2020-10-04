# Layer-wise stacked auto-encoder
This is implementation of layer-wise training of a stacked auto-encoder.


[Deep Neural Network Based Instrument Extraction From Music](http://150.162.46.34:8080/icassp2015/pdfs/0002135.pdf)

## Intro

## Environment
* Python3
``` pip install -r requirements.txt```

## How-to
Initiate the class with 

```SAE = Stacked_AEC(*save_dir*, *mode*)```

**mode** - is currently either the 'Identity' initilization with 1s across the diagonal or 'LS' for least square initialization.

```SAE.fit(train_X=tr_X, train_y=tr_y, test_X=test_X, test_y=test_y, depth=depth, epochs=epoch, lr_rate=rate, batch_size=100, opti='SGD', loss='mse')```

Pass 'Adam' for Adam optimizer or pass the whole optimizer yourself:
```
opt = RMSprop(rate)
SAE.fit(train_X=tr_X, train_y=tr_y, test_X=test_X, test_y=test_y, depth=depth, epochs=epoch, lr_rate=rate, batch_size=100, opti=opt, loss='mse')
```

## Mode
### Eye initialization
Initializes the layer weights with 1 on the main diagonal and 0s everywhere else. 
``` 
[1 0 0, 
0 1 0, 
0 0 1]
```
### Least-squares initialization
Initializes the weights based on the normalized input values (first layer with the training set, following layers with the predictions of the network). [Read more here!](http://150.162.46.34:8080/icassp2015/pdfs/0002135.pdf)



