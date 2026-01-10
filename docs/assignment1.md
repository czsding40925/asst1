# CS 248A Assignment 1 (Due Jan 26, 11:59pm)

## Environment Setup

Please read the [top level README.md](../README.md) for instructions on how to set up the environment for this assignment.

## Download 3D Models

This assignment uses several 3D models for testing and rendering.  Please download the models from [this google drive link](https://drive.google.com/drive/folders/1IHUzNzK4TwybdGrdUOpoYpKfIZEC9VVY?usp=share_link)

Place all the files under the `resources` folder in the root of the repository. Note that this folder is ignored by `.gitignore`, so if you are collaborating with your partner, please make sure both of you download the models.

## What you will do

In this assignment you will implement a basic ray caster that can render a number of different geometric representations. The assignment is divided into two parts which will be released in two steps.

In __Part 1__ of the assignment you will get acquainted with the assignment code base and learn the basics of programming in [Slang](https://shader-slang.org/), a language for writing graphics code that runs on the GPU. Specifically, in addition to a few Slang practice exercises you will implement ray-triangle intersection and a simple ray caster that generates rays based on a model of a simple camera. By the end of part 1, you will have implemented a ray caster that can render a 3D triangle mesh using a camera at a given point in the scene.

In __Part 2__ of the assignment (which will be released on Wednesday of week 2), you will extend your ray caster with a bounding volume hierarchy that will significantly increase its performance when rendering meshes with large numbers of triangles.  You will also implement ray tracing of two additional geometric representations: a voxel grid and a signed-distance function.

### Getting started on Part 1

To get started on part 1, you'll work in a Python notebook.  Please open [the part 1 notebook](../notebooks/assignment1-part1/ray-triangle-intersection.ipynb) and follow the instructions in the notebook.

The starter code also comes with an interactive renderer that allows you to move around and render 3D scenes in real time. Follow the [interactive renderer guide](interactive-renderer.md) to learn how to use it. Of course, until you implement ray-triangle intersection and camera ray casting parts of the assignment, when you press the render button in the interactive viewer you aren't going to see much!

### Grading and Handin

Assignment handin will be done on Gradescope. Instructions will be given soon.  

We will release the full grading rubric on Wed Jan 14th. All programming assignments in CS248A will be graded via a 15 minute in-person conversation with a course CA.  The CAs will ask you to render various scenes, and ask you questions about your code.  Your grade will be a function of both your ability to demonstrate correct code and your team's ability to answer CA questions about the code.
