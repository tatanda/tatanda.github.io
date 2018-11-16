---
layout: archive
permalink: /projects/
title: #"Projects by Tags"
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
{% include base_path %}
{% include group-by-array collection=site.posts field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}
