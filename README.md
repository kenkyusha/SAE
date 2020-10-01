# Layer-wise stacked auto-encoder
This is implementation of layer-wise training of a stacked auto-encoder.


[Deep Neural Network Based Instrument Extraction From Music](http://150.162.46.34:8080/icassp2015/pdfs/0002135.pdf)

## Intro

## How-to
Initiate the class with 
SAE = Stacked_AEC(*save_dir*, *mode*)
mode - is currently either the 'Identity' initilization with 1s across the diagonal or 'LS' for least square initialization.
SAE.fit(train_X=tr_X, train_y=tr_y, test_X=test_X, test_y=test_y, depth=depth, epochs=epoch, lr_rate=rate, batch_size=100)

