from spatial_manifolds.eddie_helper import run_python_script, run_stage_script

assay_mode = "GC"  # 'GC' for grid cells, 'NGS' for non grid spatial cells
mouse_days = {25: [25,24]}

for mouse, days in mouse_days.items():
    for day in days:
        data_path = f"/exports/eddie/scratch/hclark3/data/M{mouse}/D{day}/"
        stageout_dict = {
            data_path: '/exports/cmvm/datastore/sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/Harry/SpatialLocationManifolds2025/data/xgboost_GC_assay/'
        }

        job_name = f"M{mouse}D{day}_xgboost_GC"

        run_python_script(f"/home/hclark3/spatial-manifolds/scripts/eddie/run_eddie_GC_xgboost_assay.py --mouse={mouse} --day={day} --assay_mode={assay_mode} --data_path={data_path}", username="hclark3", email="hclark3@ed.ac.uk", cores=8, job_name=job_name)
        run_stage_script(stageout_dict, hold_jid=job_name)