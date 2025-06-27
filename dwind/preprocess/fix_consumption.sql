-- sage.hpc.nrel.gov
create table scratch.energy_plus_normalized_load_fixed as (
	with fix_hdf as (
	    select
	        132 as hdf_index, 
	        crb_model, 
	        nkwh
	    from load.energy_plus_normalized_load
	    where hdf_index = 133 
	    and crb_model not in ('high_energy', 'low_energy')
	    union all
	    select 
	        189 as hdf_index, 
	        crb_model,
	        nkwh
	    from load.energy_plus_normalized_load
	    where hdf_index = 191 
	    and crb_model not in ('high_energy', 'low_energy')
	    union all
	    select 
	        190 as hdf_index, 
	        crb_model, 
	        nkwh
	    from load.energy_plus_normalized_load
	    where hdf_index = 191 
	    and crb_model not in ('high_energy', 'low_energy')
	), 
	final as (
	    select 
	        hdf_index, 
	        crb_model, 
	        nkwh
	    from load.energy_plus_normalized_load
	    where crb_model not in ('high_energy', 'low_energy')
	    and hdf_index not in (132, 189, 190)
	    union all
	    select 
	        hdf_index, 
	        crb_model, 
	        nkwh
		from fix_hdf
	)
	select
	    hdf_index, 
	    crb_model, 
	    nkwh as consumption_hourly
	from final
);

select count(*) from scratch.energy_plus_normalized_load_fixed; -- 15,907

create table "load".energy_plus_normalized_load_fixed as (
	select * from scratch.energy_plus_normalized_load_fixed
);

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE "load".energy_plus_normalized_load_fixed TO dwindwrite;
GRANT SELECT ON TABLE "load".energy_plus_normalized_load_fixed TO dwindread;

GRANT USAGE ON SCHEMA "load" TO dwindread;
GRANT CREATE, USAGE ON SCHEMA "load" TO dwindwrite;