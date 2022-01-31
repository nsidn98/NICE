## NICE: Robust Scheduling through Reinforcement Learning-Guided Integer Programming

### Abstract
Integer programs provide a powerful abstraction for representing a wide range of real-world scheduling problems. Despite their ability to model general scheduling problems, solving large-scale integer programs (IP) remains a computational
challenge in practice. The incorporation of more complex
objectives such as robustness to disruptions further exacerbates the computational challenge. We present NICE (Neural network IP Coefficient Extraction), a novel technique that
combines reinforcement learning and integer programming to
tackle the problem of robust scheduling. More specifically,
NICE uses reinforcement learning to approximately represent
complex objectives in an integer programming formulation.
We use NICE to determine assignments of pilots to a flight
crew schedule so as to reduce the impact of disruptions. We
compare NICE with (1) a baseline integer programming formulation that produces a feasible crew schedule, and (2) a
robust integer programming formulation that explicitly tries
to minimize the impact of disruptions. Our experiments show
that, across a variety of scenarios, NICE produces schedules
resulting in 33% to 48% fewer disruptions than the baseline
formulation. Moreover, in more severely constrained scheduling scenarios in which the robust integer program fails to produce a schedule within 90 minutes, NICE is able to build robust schedules in less than 2 seconds on average.

<!-- ADD AUthors, affiliations and link to paper -->

You can use the [editor on GitHub](https://github.com/nsidn98/NICE/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/nsidn98/NICE/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
