# English-to-Flutter-widget

In this repository we'll train **Seq2seq Transformer** model to convert natural English sentences into Flutter UI widget code.

### Note
The goal of this experiment is totally for research purposes, We're not going to support all Flutter widgets for the current phase and also we are using some custom Widget instead of the regular widget to make things easier


## Dataset
Our generated dataset has around 175000 English sentences and around 175000 Flutter widget code

The Dataset is a Josn file containing array of object, each object has two properties, "sentence" and "widget".
The "sentence" property is the English sentence
and the "widget" property is the Flutter code.


```
{
    "widget": "Container ( width: value ) ",
     "sentence": "create a container with width value"
}
```

Why do we have “value” instead of numbers since the property is “Width”?
Since we have a lot of dynamic properties values, like Colors, width, height, padding, alignment, etc…
We are going to do some preprocessing for our input to change all of these values with keyword “Value” and will do post processing to the output of the model to get the real values back.
So it will be something like this:
“Create a container with width 24.0” => “create a container with width value”.

## Tokenization

For English sentences, we’re going to use “Spacy” as our input Tokenizer For the output Tokenizer, we’ll build our own custom tokenizer. We didn’t find a suitable tokenizer for Dart/Flutter so we’ll be using Python’s default tokenize for now and it will be changed later on.

We’ll be using PyTorch’s torchtext.data.Field:
For the input we’ll use Spacy as mentioned, which it implemented by default in data.Field

```
Input = Field(tokenize = 'spacy',
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)
```

```
from tokenize import tokenize, untokenize
import io
def tokenize_flutter_code(str_code):
    tokens = list(tokenize(io.BytesIO(str_code.encode('utf-8')).readline))
    return [it.string for it in flutter_tokens]
```
We’ll call it throw the Field of output:

```
Output = Field(tokeniz = tokenize_flutter_code ,
               init_token = '<sos>',
               eos_token = '<eos>',
               lower = False)
```

## Sample results
Create a text with "this is my text" inside row
```
Row(children:[Text("this is my text")],)
```
Build a box with color Color.red
```
Container(color:Colors.red)
```
Draw a box with width 24
```
Container(width:24)
```
Add a text with "Hello" and textSize 22.0
```
CustomText("Hello",fontSize:22.0)
```
Build a box with width 450 and height 450 and color Colors.red inside center
```
Center(child:Container(width:450, height:450, color: Colors.red))
```
Center parent of text with "This is text inside a center widget"
```
Center(child:Text("This is text inside a center widget"))
```
get an Image with "img.png"
```
Image.asset("img.png")
```
get network Image with "http://example.png"
```
Image.network("http://example.png")
```

