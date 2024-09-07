# Decision-Tree-from-Scratch
Decision Tree through ID3 algorithm

## Performance:

### Titanic Dataset:

```console
Train Performance: 87%
Test Performance: 85%
```

## Tree-Representation:

```console
`->Cough==NO
   `->Fever==NO
      `->Breathing issues==NO
         `->{'NO': '100.0%'}
   `->Fever==YES
      `->Breathing issues==YES
         `->{'YES': '100.0%'}
`->Cough==YES
   `->Fever==NO
      `->Breathing issues==NO
         `->{'NO': '100.0%'}
      `->Breathing issues==YES
         `->{'NO': '33.3%', 'YES': '66.7%'}
   `->Fever==YES
      `->Breathing issues==NO
         `->{'NO': '66.7%', 'YES': '33.3%'}
      `->Breathing issues==YES
         `->{'YES': '100.0%'}
```
### Cardiac Dataset:
```
Train Performance: 83%
Test Performance: 100%
```
