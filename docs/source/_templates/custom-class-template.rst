{{ fullname | escape | underline}}

{% set route = pystra_api_routes.get(fullname.split('.')[1]) %}
{% if route %}
**Use it:** :doc:`User guide </{{ route[0] }}>` · :doc:`Worked example </notebooks/{{ route[1] }}>` · :doc:`Theory </theory/{{ route[2] }}>`
{% endif %}

.. currentmodule:: {{ module }}

{% if fullname == 'pystra.reliability.form.FORM' %}
.. autoclass:: {{ objname }}
   :show-inheritance:

Everyday operations
-------------------

.. automethod:: FORM.run

{% else %}
.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :inherited-members:
   :special-members: __call__, __add__, __mul__

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
   {% for item in methods %}
      {%- if not item.startswith('_') %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% endblock %}

{% endif %}
