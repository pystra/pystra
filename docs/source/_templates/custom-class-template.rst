{{ fullname | escape | underline}}

{% set route = pystra_api_routes.get(fullname.split('.')[1]) %}
{% if route %}
**Use it:** :doc:`User guide </{{ route[0] }}>` · :doc:`Worked example </notebooks/{{ route[1] }}>` · :doc:`Theory </theory/{{ route[2] }}>`
{% endif %}

.. currentmodule:: {{ module }}

{% if fullname == 'pystra.form.Form' %}
.. autoclass:: {{ objname }}
   :show-inheritance:

Everyday operations
-------------------

.. automethod:: Form.run

.. automethod:: Form.get_failure

.. automethod:: Form.get_beta

.. automethod:: Form.get_equivalent_beta

.. automethod:: Form.get_design_point

.. automethod:: Form.get_alpha

.. automethod:: Form.get_no_function_calls

.. automethod:: Form.show_results

.. automethod:: Form.show_detailed_output

Algorithm extension methods
---------------------------

These stages support implementation and extension of the design-point search.
Use ``run()`` for an ordinary analysis so initialization and convergence checks
are performed in the intended order.

{% for item in methods %}
{% if not item.startswith("_") and item not in ['run', 'get_failure', 'get_beta', 'get_equivalent_beta', 'get_design_point', 'get_alpha', 'get_no_function_calls', 'show_results', 'show_detailed_output'] %}
.. automethod:: Form.{{ item }}

{% endif %}
{% endfor %}
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
