# Modified from:
# https://github.com/simonsobs/pwg-scripts/blob/844e278384d9ce6d505585b2e7cb7cf3acf45a43/pwg-tds/sim-sso/write_context.py#L3

"""write_context.py -- create context.yaml

This script indexes output in TOAST's native HDF5 format.  You can
index any number of output directories and it will figure out what's
up with the tubes and telescopes and so on.

"""
import glob
import os

import h5py
import numpy as np
import sotodlib.toast as sotoast
import yaml
from so3g.proj import quat
from sotodlib.core import metadata
from sotodlib.io.metadata import read_dataset, write_dataset

WAFER_CACHE = {}


def get_wafer_info(telescope, cache_file=None):
    if telescope not in WAFER_CACHE and cache_file is not None:
        try:
            WAFER_CACHE[telescope] = read_dataset(cache_file, telescope)
        except:
            print("Failed to load from cache file.")
    if telescope not in WAFER_CACHE:
        focalplane = sotoast.SOFocalplane(telescope=telescope)
        # Reduced info ...
        subkeys = ["band", "tube_slot", "wafer_slot"]
        subitems = set()
        for k in focalplane.keys():
            subitems.add(tuple([focalplane[k][_k] for _k in subkeys]))
        wafers = metadata.ResultSet(keys=subkeys)
        wafers.rows = sorted(list(subitems))
        WAFER_CACHE[telescope] = wafers
        if cache_file is not None:
            write_dataset(wafers, cache_file, telescope)
    return WAFER_CACHE[telescope]


tube_types = {
    "f030": "LF",
    "f040": "LF",
    "f090": "MF",
    "f150": "MF",
    "f230": "UHF",
    "f290": "UHF",
}


def guess_tube(telescope, wafers):
    if telescope == "SAT":
        for t in ["SAT1", "SAT2", "SAT3", "SAT4"]:
            try:
                return guess_tube(t, wafers)
            except AssertionError:
                pass
        assert False  # Did not found wafer in any SAT?

    info = get_wafer_info(telescope, "telcache")
    s = np.zeros(len(info), bool)
    for w in wafers:
        s += info["wafer_slot"] == w

    info = info.subset(keys=["tube_slot", "wafer_slot"], rows=s).distinct()
    assert len(set(info["tube_slot"])) == 1  # only one tube_slot!

    slot_mask = ""
    for w in sorted(list(set(info["wafer_slot"]))):
        slot_mask += "1" if w in wafers else "0"

    return telescope, info["tube_slot"][0], slot_mask


def extract_detdb(hg, db=None):
    if db is None:
        db = metadata.DetDb()
        db.create_table(
            "base",
            [
                "`det_id_` text",  # we can't use "det_id"; rename later
                "`readout_id` text",
                "`wafer_slot` text",
                "`special_ID` text",
                "`tel_type` text",
                "`tube_type` text",
                "`band` text",
                "`fcode` text",
                "`toast_band` text",
            ],
        )
        db.create_table(
            "quat",
            [
                "`r` float",
                "`i` float",
                "`j` float",
                "`k` float",
            ],
        )

    existing = list(db.dets()["name"])

    tel_type = hg["instrument"].attrs.get("telescope_name")
    if tel_type in ["LAT", "SAT"]:
        pass
    else:
        # new toast has TELn_TUBE
        tel_type = tel_type.split("_")[0][:3]
    assert tel_type in ["LAT", "SAT"]

    fp = hg["instrument"]["focalplane"]
    for dv in fp:
        v = dict(
            [(_k, dv[_k].decode("ascii")) for _k in ["wafer_slot", "band", "name"]]
        )
        k = v.pop("name")
        if k in existing:
            continue
        v["special_ID"] = int(dv["uid"])
        v["toast_band"] = v["band"]
        v["band"] = v["toast_band"].split("_")[1]
        v["fcode"] = v["band"]
        v["tel_type"] = tel_type
        v["tube_type"] = tube_types[v["band"]]
        v["det_id_"] = "DET_" + k
        v["readout_id"] = k
        db.add_props("base", k, **v, commit=False)
        db.add_props(
            "quat",
            k,
            **{
                "r": dv["quat"][3],
                "i": dv["quat"][0],
                "j": dv["quat"][1],
                "k": dv["quat"][2],
            },
        )

    db.conn.commit()
    db.validate()
    return db


def extract_obs_info(h):
    t = np.asarray(h["shared"]["times"])[[0, -1]]
    data = {
        "toast_obs_name": h.attrs["observation_name"],
        "toast_obs_uid": int(h.attrs["observation_uid"]),
        "target": h.attrs["observation_name"].split("-")[0].lower(),
        "start_time": t[0],
        "stop_time": t[1],
        "timestamp": t[0],
        "duration": t[1] - t[0],
    }
    return data


def detdb_to_focalplane(db):
    # Focalplane compatible with, like, planet mapper.
    fp = metadata.ResultSet(keys=["dets:readout_id", "xi", "eta", "gamma"])
    for row in db.props(
        props=["readout_id", "quat.r", "quat.i", "quat.j", "quat.k"]
    ).rows:
        q = quat.quat(*row[1:])
        xi, eta, gamma = quat.decompose_xieta(q)
        fp.rows.append((row[0], xi, eta, (gamma) % (2 * np.pi)))
    return fp


def create_context(context_dir, export_dirs, absolute=False):
    if not os.path.exists(context_dir):
        os.makedirs(context_dir, exist_ok=True)

    tel_info_cachefile = os.path.join(context_dir, "tels.h5")

    obsfiledb = metadata.ObsFileDb()
    obsdb = metadata.ObsDb()
    obsdb.add_obs_columns(
        [
            "timestamp float",
            "duration float",
            "start_time float",
            "stop_time float",
            "telescope string",
            "tel_tube string",
            "wafer_slots string",
            "target string",
            "toast_obs_name string",
            "toast_obs_uid string",
        ]
    )

    detsets = {}
    
    # For single data dir...
    for export_dir in export_dirs:
        print(export_dir)
        files = glob.glob(os.path.join(export_dir, "*h5"))
        print(f"Found {len(files)} files in {export_dir}")

        for filename in files:
            with h5py.File(filename, "r") as h:
                detdb = extract_detdb(h, db=None)
                obs_info = extract_obs_info(h)

            # Convert detdb to ResultSet
            props = detdb.props()
            props.keys[props.keys.index("det_id_")] = "det_id"  

            # Figure out a couple more things ...
            tel_type = props["tel_type"][0]
            wafers = set(props["wafer_slot"])
            telescope, tube, slot_mask = guess_tube(tel_type, wafers)
            print(slot_mask)
            bands = list(set(props["band"]))

            # Merge in telescope name
            props.merge(
                metadata.ResultSet(keys=["telescope"], src=[(telescope,)] * len(props))
            )

            obs_info.update(
                {
                    "telescope": telescope,
                    "tel_tube": tube,
                    "wafer_slots": ",".join(wafers),
                }
            )

            # In this format, all dets for the set of wafers and a
            # single band are stored in one file.  Create that detset
            # name from the list of wafers + band(s).
            detset = "_".join(sorted(list(wafers)) + sorted(list(bands)))
            if detset not in detsets:
                fp = detdb_to_focalplane(detdb)
                detsets[detset] = [props, fp]
                obsfiledb.add_detset(detset, props["readout_id"])

            assert len(bands) == 1
            obs_id = f'{int(obs_info["timestamp"])}_{tube}_{bands[0]}_{slot_mask}'
            obsfiledb.add_obsfile(
                #os.path.split(filename)[1],
                os.path.basename(filename),
                obs_id,
                detset,
                0,
                1,
            )
            # obs info
            obsdb.update_obs(obs_id, obs_info)
            print(f"  added {obs_id}")
    #TODO: above here
            
    # detdb.to_file(f'{context_dir}/detdb.sqlite')
    obsdb.to_file(f"{context_dir}/obsdb.sqlite")
    obsfiledb.to_file(f"{context_dir}/obsfiledb.sqlite")

    #
    # metadata: det_info & focalplane
    #

    scheme = metadata.ManifestScheme()
    scheme.add_exact_match("dets:detset")
    scheme.add_data_field("dataset")
    db1 = metadata.ManifestDb(scheme=scheme)

    scheme = metadata.ManifestScheme()
    scheme.add_exact_match("dets:detset")
    scheme.add_data_field("dataset")
    db2 = metadata.ManifestDb(scheme=scheme)

    for detset, (props, fp) in detsets.items():
        key = "dets_" + detset
        props.keys = ["dets:" + k for k in props.keys]
        write_dataset(props, f"{context_dir}/metadata.h5", key, overwrite=True)
        db1.add_entry({"dets:detset": detset, "dataset": key}, filename="metadata.h5")

        key = "focalplane_" + detset
        write_dataset(fp, f"{context_dir}/metadata.h5", key, overwrite=True)
        db2.add_entry({"dets:detset": detset, "dataset": key}, filename="metadata.h5")

    db1.to_file(f"{context_dir}/det_info.sqlite")
    db2.to_file(f"{context_dir}/focalplane.sqlite")

    # And the context.yaml!
    context = {
        "tags": {"metadata_lib": "./"},
        "imports": ["sotodlib.io.metadata"],
        "obsfiledb": "{metadata_lib}/obsfiledb.sqlite",
        #'detdb': '{metadata_lib}/detdb.sqlite',
        "obsdb": "{metadata_lib}/obsdb.sqlite",
        "obs_loader_type": "toast3-hdf",
        "obs_colon_tags": ["wafer_slot", "band"],
        "metadata": [
            {"db": "{metadata_lib}/det_info.sqlite", "det_info": True},
            {"db": "{metadata_lib}/focalplane.sqlite", "name": "focal_plane"},
        ],
    }
    
    if absolute:
        context["tags"]["metadata_lib"] = context_dir

    open(f"{context_dir}/context.yaml", "w").write(
        yaml.dump(context, sort_keys=False)
    )
