As originally seen in [https://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html]. 
This demo learns a mapping from (x,y) in the image to (r,g,b) with a network
with limited capacity and then uses it to reconstruct the image.

Run 
    
    ./image_reco.py image exp

Arguments: image is an image; exp defines the name of the temporary files
that get generated, so you can run multiple images. 
