export default function ParameterField({ parameter, value, onChange, error, disabled }) {
  const common = { id: `parameter_${parameter.name}`, disabled, 'aria-invalid': !!error, 'aria-describedby': `help_${parameter.name}` };
  return <div className={`parameter_field ${error ? 'invalid' : ''}`}>
    <label htmlFor={common.id}>{parameter.label}{parameter.required && <em> *</em>}</label>
    {parameter.type === 'boolean' ? <label className="switch_row"><input {...common} type="checkbox" checked={!!value} onChange={event => onChange(event.target.checked)} /><span>{value ? '已开启' : '未开启'}</span></label>
      : parameter.type === 'select' ? <select {...common} value={value} onChange={event => onChange(event.target.value)}><option value="">请选择</option>{parameter.options.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}</select>
      : parameter.type === 'textarea' ? <textarea {...common} rows={3} value={value} maxLength={parameter.max_length ?? 2000} onChange={event => onChange(event.target.value)} />
      : <input {...common} type={['number', 'integer'].includes(parameter.type) ? 'number' : 'text'} value={value} min={parameter.min} max={parameter.max} step={parameter.type === 'integer' ? 1 : parameter.step ?? 'any'} maxLength={parameter.max_length ?? 2000} onChange={event => onChange(event.target.value)} />}
    <small id={`help_${parameter.name}`} className={error ? 'field_error' : ''}>{error || parameter.help || (parameter.min != null || parameter.max != null ? `范围 ${parameter.min ?? '不限'} – ${parameter.max ?? '不限'}` : '')}</small>
  </div>;
}
