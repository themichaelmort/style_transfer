

<p align="center">
  <img width=75% src="https://github.com/themichaelmort/style_transfer/blob/main/style_transfer.gif" alt="Time-lapse of style transfer"> 
</p>
<p align="center">
  Time lapse of a photograph of a pumpkin as the style of Van Gogh's Starry Night is transferred.
</p>

<h1>Style Transfer</h1>

<p>
  I am not much of a painter myself, but I love the way some painters use texture to develop a personal style for their art. For this project, I used a technique called style transfer, introduced by Gatys et al. in their paper <a href="https://arxiv.org/pdf/1508.06576.pdf">“A Neural Algorithm of Artistic Style”</a> to transform a photograph of a pumpkin so that it looked like it was a picture of a pumpkin painted by Vincent Van Gogh in the style of his famous master work <a href="https://en.wikipedia.org/wiki/The_Starry_Night">“The Starry Night”</a>. Mathematically, this project amounted to projecting the pumpkin image into the style space of the “Starry Night” with a <a href="https://en.wikipedia.org/wiki/Gram_matrix">Gram Matrix</a>.
</p>

<h2>At a Glance</h2>

<ul>
  <li>Models - Style transfer uses two models, one for content and one for style. In this project, both models were pretrained VGG-16 model from <a href="https://arxiv.org/abs/1409.1556">"Very Deep Convolutional Networks for Large-Scale Image Recognition"</a></li> as implemented and stored in PyTorch. The models were set in inference mode.
  <li>Optimizer - Adam, as implemented in PyTorch</li>
  <li>Loss function - The loss function consists of two parts: 1.) Content loss - which tries to minimize the difference between the generated image and the original image (thus encouraging the algorithm to preserve the content), and 2.) Style loss - which tries to minimize the difference between the style of the generated image and the style of the style image. These two loss values are weighted against each other. See <a href="https://arxiv.org/pdf/1508.06576.pdf">Gatys et al.</a> for more details.</li>
    <ul>
      <li>Note: The script style_transfer.py was drafted in Google Colab. If it is deployed elsewhere an different image loading API may be necessary to load content and style images.</li>
    </ul>
</ul>

<h3> Content Image </h3>
<p>
  <img width=50% src="https://github.com/themichaelmort/style_transfer/blob/main/content_image.jpg" alt="Time-lapse of style transfer"> 
</p>

<h3>Style Image</h3>
<p>
  <img width=50% src="https://github.com/themichaelmort/style_transfer/blob/main/style_image.jpg" alt="Time-lapse of style transfer"> 
</p>

<h2>Remarks</h2>
It can be difficult to separate the style of a picture from its content. Indeed, content and style are as much a matter of human perception than anything. Thus, the results of style transfer are dependent on the desired aesthetic of the user. The weights used in the loss function were hand tuned for this specific style transfer task, and would likely require retuning for another content or style image.
