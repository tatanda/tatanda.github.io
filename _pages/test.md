---
layout: categories
permalink: #/test/
title: "Posts by Category"
author_profile: true
header:
  image: #""
scope:
  path: ""
  type: pages
values:
  layout: single
  #author_profile: false
---
<h1>Latest Posts</h1>

<ul>
  {% for post in site.posts %}
    <li>
      <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
      <p>{{ post.excerpt }}</p>
    </li>
  {% endfor %}
</ul>
