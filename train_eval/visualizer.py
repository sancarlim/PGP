import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from typing import Dict, List
from train_eval.initialization import initialize_prediction_model, initialize_dataset, get_specific_args
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction.input_representation.static_layers_original import StaticLayerRasterizer, color_by_yaw
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface_original import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer
from nuscenes.prediction.helper import convert_local_coords_to_global, convert_global_coords_to_local
import train_eval.utils as u
from train_eval.utils import Collate_heterograph
import imageio
import os 
import scipy.sparse as spp
from nuscenes.map_expansion.map_api import NuScenesMap
from scipy.ndimage import rotate
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw 
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import math
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ego_car = plt.imread('/media/14TBDISK/sandra//DBU_Graph/NuScenes/icons/Car TOP_VIEW ROBOT.png')
agent = plt.imread('/media/14TBDISK/sandra/DBU_Graph/NuScenes/icons/Car TOP_VIEW 375397.png')
cars = plt.imread('/media/14TBDISK/sandra/DBU_Graph/NuScenes/icons/Car TOP_VIEW 80CBE5.png') 

layers = ['drivable_area',  
          #'lane', 
          #'road_segment', 
          #'road_block', 
          'ped_crossing',
          'walkway',  ]


class Visualizer:
    """
    Class for visualizing predictions generated by trained model
    """
    def __init__(self, cfg: Dict, data_root: str, data_dir: str, checkpoint_path: str, examples: int, frames: List[int],
                    show_preds: bool = True, tf: int = 12, num_modes: int = 10, counterfactual: bool = False, 
                    mask_lanes = False, name: str = ''):
        """
        Initialize evaluator object
        :param cfg: Configuration parameters
        :param data_root: Root directory with data
        :param data_dir: Directory with extracted, pre-processed data
        :param checkpoint_path: Path to checkpoint with trained weights
        """

        # Initialize dataset
        ds_type = cfg['dataset'] + '_' + cfg['agent_setting'] + '_' + cfg['input_representation']
        spec_args = get_specific_args(cfg['dataset'], data_root, cfg['version'] if 'version' in cfg.keys() else None)
        test_set = initialize_dataset(ds_type, ['load_data', data_dir, cfg['test_set_args']] + spec_args[0])
        self.ds = test_set
        self.encoder_type = cfg['encoder_type']
        if 'scout' in cfg['encoder_type']:
            self.collate_fn = Collate_heterograph(cfg['encoder_args'])
        self.lane_mask_prob = cfg['encoder_args']['lane_mask_prob']
        self.agent_mask_prob_v = cfg['encoder_args']['agent_mask_prob_v']
        self.mask_frames = cfg['encoder_args']['mask_frames']
        self.ns = spec_args[1][0]
        self.dataroot = data_root
        self.examples = examples
        self.frames = frames
        self.show_predictions = show_preds
        self.tf = tf*2 # 2 frames per second
        self.num_modes_to_show = num_modes
        self.counterfactual = counterfactual
        self.mask_lanes = mask_lanes
        self.name = name
        self.legend = False
        self.patch_margin = 25
        self.min_diff_patch = 25

        # Initialize model
        self.model = initialize_prediction_model(cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
                                                 cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'])
        self.model = self.model.float().to(device)
        self.model.eval()

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])


        # Raster maps for visualization.
        map_extent = self.ds.map_extent
        resolution = 0.1
        static_layer_rasterizer = StaticLayerRasterizer(self.ds.helper,
                                                        resolution=resolution,
                                                        meters_ahead=map_extent[3],
                                                        meters_behind=-map_extent[2],
                                                        meters_left=-map_extent[0],
                                                        meters_right=map_extent[1])

        agent_rasterizer = AgentBoxesWithFadedHistory(self.ds.helper, seconds_of_history=1,
                                                      resolution=resolution,
                                                      meters_ahead=map_extent[3],
                                                      meters_behind=-map_extent[2],
                                                      meters_left=-map_extent[0],
                                                      meters_right=map_extent[1])

        self.raster_maps = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

    def visualize(self, output_dir: str, dataset_type: str):
        """
        Generate visualizations for predictions
        :param output_dir: results directory to dump visualizations in
        :param dataset_type: e.g. 'nuScenes'. Visualizations will vary based on dataset.
        :return:
        """
        if dataset_type == 'nuScenes':
            self.visualize_nuscenes(output_dir)

    def visualize_nuscenes(self, output_dir):
        index_list = self.get_vis_idcs_nuscenes()
        if not os.path.isdir(os.path.join(output_dir, 'results', 'gifs')):
            os.mkdir(os.path.join(output_dir, 'results', 'gifs'))
        start = time.time()
        indexes_list = [index_list[i] for i in self.examples]
        for n, indices in enumerate(indexes_list):
            self.example = self.examples[n]
            if self.frames is not None:
                indices = [indices[self.frames[n]]]  
            fancy_img, graph_img, scene = self.generate_nuscenes_gif(indices)
            filename = os.path.join(output_dir, 'example_' + str(self.example) + self.name + scene + '.gif')
            imageio.mimsave(filename, fancy_img, format='GIF', fps=2)  
            for i,img in enumerate(fancy_img):
                filename = os.path.join(output_dir, 'example' + str(self.example) + '_' + self.name + scene  + '_' + str(self.frames[n]) +'.png')
                plt.imsave(filename, img)  
            filename = os.path.join(output_dir, 'results', 'gifs', 'example' + str(self.example)+ scene + '_graph_' + self.name + '.gif')
            imageio.mimsave(filename, graph_img, format='GIF', fps=2)  
            
            print('Saved gif for example ' + str(self.example) + ' in ' + str(time.time() - start) + ' seconds')

    def get_vis_idcs_nuscenes(self):
        """
        Returns list of list of indices for generating gifs for the nuScenes val set.
        Instance tokens are hardcoded right now. I'll fix this later (TODO)
        """
        token_list = get_prediction_challenge_split('val', dataroot=self.ds.helper.data.dataroot)
        instance_tokens = [token_list[idx].split("_")[0] for idx in range(len(token_list))]
        unique_instance_tokens = []
        for i_t in instance_tokens:
            if i_t not in unique_instance_tokens:
                unique_instance_tokens.append(i_t)

        instance_tokens_to_visualize = [54, 98, 91, 5, 114, 144, 291, 204, 312, 187, 36, 267, 146,
                                        56,82,89,93,109,111,113,127,166,104]

        idcs = []
        for i_t_id in instance_tokens_to_visualize:
            idcs_i_t = [i for i in range(len(instance_tokens)) if instance_tokens[i] == unique_instance_tokens[i_t_id]]
            idcs.append(idcs_i_t)

        return idcs

    def visualize_graph(self, fig, ax, node_feats, s_next, edge_type, evf_gt, node_seq, fut_xy, pi, cmap_cool, 
                        vehicles_feats, graph, att):
        """
        Function to visualize lane graph.
        """ 
        ax.imshow(np.zeros((3, 3)), extent=[-60,60,-40,100], cmap='gist_gray') 
        
        # Plot lane edges - successors and proximals
        for src_id, src_feats in enumerate(node_feats):
            feat_len = np.sum(np.sum(np.absolute(src_feats), axis=1) != 0)
            # Convert to global feats  
            if feat_len > 0:
                src_x = np.mean(src_feats[:feat_len, 0])
                src_y = np.mean(src_feats[:feat_len, 1])

                for idx, dest_id in enumerate(s_next[src_id]):
                    edge_t = edge_type[src_id, idx]
                    visited = evf_gt[src_id, idx]
                    pi_edge = np.exp(pi[src_id, idx])
                    if 3 > edge_t > 0: 
                        dest_feats = node_feats[int(dest_id)]
                        feat_len_dest = np.sum(np.sum(np.absolute(dest_feats), axis=1) != 0)
                        dest_x = np.mean(dest_feats[:feat_len_dest, 0])
                        dest_y = np.mean(dest_feats[:feat_len_dest, 1])
                        d_x = dest_x - src_x
                        d_y = dest_y - src_y

                        line_style = '-' if edge_t == 1 else '--'
                        color = cmap_cool(pi_edge) if edge_t == 1 else 'black'
                        width = .5 if visited else 0.005
                        alpha = 1 if visited else 0.5

                        plt.arrow(src_x, src_y, d_x, d_y, color=color, head_width=0.1, length_includes_head=True,
                                  linestyle=line_style, width=width, alpha=alpha)

        # Plot lane nodes
        lane_pos = []
        for node_id, node_feat in enumerate(node_feats):
            feat_len = np.sum(np.sum(np.absolute(node_feat), axis=1) != 0)
            if feat_len > 0:
                visited = node_id in node_seq
                x = np.mean(node_feat[:feat_len, 0])
                y = np.mean(node_feat[:feat_len, 1])
                lane_pos.append([x, y])
                yaw = np.arctan2(np.mean(np.sin(node_feat[:feat_len, 2])),
                                 np.mean(np.cos(node_feat[:feat_len, 2])))
                c = color_by_yaw(0, yaw)
                c = np.asarray(c).reshape(-1, 3) / 255
                s = 70 if visited else 30
                ax.scatter(x, y, s, c=c)
            else:
                lane_pos.append([0, 0])
        
        #Plot surrounding vehicles nodes 
        for node_id, node_feat in enumerate(vehicles_feats): 
            if node_feat.sum() == 0:
                break 
            if node_id == 0:
                c = 'red'
                markersize = 10
            else:
                c = 'blue'
                markersize = 7 
            plt.plot(node_feat[-1, 0], node_feat[-1, 1], '*', color=c, markersize=markersize)
            ax.annotate(str(node_id), (node_feat[-1, 0], node_feat[-1, 1]), color='w', fontsize=10)
        
        # Retrieve object-level edge attention -> att[3:]. att[:3] is type-level attention.
        v2v_attn = att[-1]
        v2l_attn = att[-2]  
        # Plot interactions with other agents for the focal vehicle
        if len(v2v_attn.shape) > 0:
            for idx, v in enumerate(graph['v_interact_v'].edges()[0]):
                # if v > graph['v_interact_v'].edges()[1][idx]:
                if graph['v_interact_v'].edges()[0][idx] == 0:
                    color = 'w'
                    # Compute attention taking into account object-level and type-level attention
                    attn = v2v_attn[idx]*att[2][-1][graph['v_interact_v'].edges()[1][idx]]*2.5 
                    w = attn
                    alpha = max(1., v2v_attn[idx]) 
                    plt.arrow(vehicles_feats[v][-1,0], vehicles_feats[v][-1,1], vehicles_feats[graph['v_interact_v'].edges()[1][idx]][-1,0]-vehicles_feats[v][-1,0], 
                        vehicles_feats[graph['v_interact_v'].edges()[1][idx]][-1,1]-vehicles_feats[v][-1,1], color=color, head_width=0.1, 
                        length_includes_head=True,  width=w, alpha=alpha) 

        # Vehicle-lane visualization
        if True:
            for idx, v in enumerate(graph['v_close_l'].edges()[0]):
                if v == 0:
                    attn = v2l_attn[idx]#*att[0][-1][graph['v_close_l'].edges()[1][idx]]
                    w = attn * 0.3 if attn > 0.6 else 0
                    alpha = max(.5, v2l_attn[idx]) 
                    plt.arrow(vehicles_feats[v][-1,0], vehicles_feats[v][-1,1], lane_pos[graph['v_close_l'].edges()[1][idx]][0]-vehicles_feats[v][-1,0], 
                        lane_pos[graph['v_close_l'].edges()[1][idx]][1]-vehicles_feats[v][-1,1], color='lightgreen', head_width=0.1, 
                        length_includes_head=True,  width=w, alpha=alpha)

        plt.show()
        return fig, ax


    def generate_nuscenes_gif(self, idcs: List[int]):
        """
        Generates gif of predictions for the given set of indices.
        :param idcs: val set indices corresponding to a particular instance token.
        """
        s_t = self.ds[idcs[0]]['inputs']['sample_token']
        scene=self.ns.get('scene', self.ns.get('sample',s_t)['scene_token'])
        scene_name=scene['name']
        log=self.ns.get('log', scene['log_token'])
        location = log['location']
        nusc_map = NuScenesMap(dataroot=self.dataroot, map_name=location)
        
        imgs_fancy = []
        graph_img = [] 
        vehicle_masked_t = []
        for idx in idcs: 
            # Load data
            data = self.ds[idx]
            i_t = data['inputs']['instance_token']
            s_t = data['inputs']['sample_token']
            annotations = self.ds.helper.get_annotations_for_sample(s_t)
            past = self.ds.helper.get_past_for_sample(s_t,2,False)
            future = self.ds.helper.get_future_for_sample(s_t,self.tf/2,False) 

            sample_record = self.ns.get('sample', s_t)
            sample_data_record = self.ns.get('sample_data', sample_record['data']['LIDAR_TOP'])
            pose_record = self.ns.get('ego_pose', sample_data_record['ego_pose_token'])
            ego_poses=np.array(pose_record['translation'][:2])
            
            # Render the map patch with the current ego poses. 
            if self.example == 4 or self.example==9:
                self.patch_margin = 25
            min_patch = np.floor(ego_poses - self.patch_margin)
            max_patch = np.ceil(ego_poses + self.patch_margin)
            diff_patch = max_patch - min_patch
            if any(diff_patch < self.min_diff_patch):
                center_patch = (min_patch + max_patch) / 2
                diff_patch = np.maximum(diff_patch, self.min_diff_patch)
                min_patch = center_patch - diff_patch / 2
                max_patch = center_patch + diff_patch / 2
            my_patch = (min_patch[0], min_patch[1], max_patch[0], max_patch[1])

            # For article visualizations
            if self.example == 5:
                my_patch = (min_patch[0]+25, min_patch[1]-30, max_patch[0]+25, max_patch[1]-30)
            elif self.example == 4:
                my_patch = (min_patch[0]+25, min_patch[1]-30, max_patch[0]+25, max_patch[1]-30)
            elif self.example == 3:  
                my_patch = (min_patch[0]-5, min_patch[1]+25, max_patch[0]-5 , max_patch[1]+25)
            elif self.example == 2:  
                my_patch = (min_patch[0]-10, min_patch[1]+10, max_patch[0]-10 , max_patch[1]+10)
            elif self.example == 13:  
                my_patch = (min_patch[0]-20, min_patch[1]+20, max_patch[0]-20 , max_patch[1]+20)
            elif self.example == 9:
                my_patch = (min_patch[0]-20, min_patch[1]-25, max_patch[0]-20 , max_patch[1]-25)
            elif self.example == 20 or self.example == 1 or self.example == 15:
                my_patch = (min_patch[0]-20, min_patch[1]+10, max_patch[0]-20 , max_patch[1]+10)

            fig2, ax2 = nusc_map.render_map_patch(my_patch, layers, figsize=(10, 10), alpha=0.3,
                                        render_egoposes_range=False,
                                        render_legend=self.legend, bitmap=None) 

            cmap_cool = plt.get_cmap('autumn')
            sm_cool = plt.cm.ScalarMappable(cmap=cmap_cool, norm=plt.Normalize(vmin=0, vmax=1)) 
            #divider = make_axes_locatable(ax2)
            #cax = divider.append_axes("right", size="2%", pad=0.7)
            cbar_cool = plt.colorbar(sm_cool, shrink=0.8)
            cbar_cool.set_label('Probability of each mode', rotation=270, labelpad=20)
                                        
            r_img = rotate(ego_car, quaternion_yaw(Quaternion(pose_record['rotation']))*180/math.pi,reshape=True)
            oi = OffsetImage(r_img, zoom=0.015, zorder=500)
            veh_box = AnnotationBbox(oi, (ego_poses[0], ego_poses[1]), frameon=False)
            veh_box.zorder = 500
            ax2.add_artist(veh_box)

            # mask out vehicles of interest
            if self.example == 2 and self.counterfactual: #idx == 0 and
                vehicle_masked_t = [annotations[4]['instance_token']] #[annotations[i]['instance_token'] for i in range(len(annotations)) if 'vehicle' in annotations[i]['category_name'] and annotations[i]['instance_token']!=i_t and mask_out[i]]
                mask_vehicles = []
                for i in range(len(annotations)):
                    if 'vehicle' in annotations[i]['category_name'] and annotations[i]['instance_token']!=i_t:
                        if annotations[i]['instance_token'] in vehicle_masked_t:
                            mask_vehicles.append(1)
                        else:
                            mask_vehicles.append(0)  
                #[mask_out[i] for i in range(len(annotations)) if 'vehicle' in annotations[i]['category_name'] and annotations[i]['instance_token']!=i_t]
            else:
                mask_vehicles = [1 if annotations[i]['instance_token'] in vehicle_masked_t else 0 for i in range(len(annotations)) if 'vehicle' in annotations[i]['category_name'] and annotations[i]['instance_token']!=i_t]
            data['inputs']['surrounding_agent_representation']['vehicle_masks'][:len(mask_vehicles)] += np.tile(np.expand_dims(np.array(mask_vehicles), (-2,-1)), [1,5,5])
            data['inputs']['agent_node_masks']['vehicles'][:,:len(mask_vehicles)] += np.tile(np.expand_dims(np.array(mask_vehicles), (0)), [164,1])  
            # Remove agent to adjacency matrix  
            data['inputs']['surrounding_agent_representation']['adj_matrix'][0,np.where(np.array(mask_vehicles)==1)[0]+1] = 0
            data['inputs']['surrounding_agent_representation']['adj_matrix'][np.where(np.array(mask_vehicles)==1)[0]+1,0] = 0
            data['inputs']['surrounding_agent_representation']['adj_matrix'][np.where(np.array(mask_vehicles)==1)[0]+1,np.where(np.array(mask_vehicles)==1)[0]+1] = 0
            
            for n, ann in enumerate(annotations):
                if ann['instance_token'] in vehicle_masked_t and ann['instance_token']!=i_t:
                    continue 
                #Plot history
                if len(past[ann['instance_token']]) > 0:
                    history =  np.concatenate((past[ann['instance_token']][::-1], np.array([ann['translation'][:2]])))
                else:
                    history = np.array([ann['translation'][:2]])
                
                if False:  #self.example == 5:
                    # Vehicle n=0 is stopped 
                    veh_token = ann['instance_token']
                    history_fict = future[i_t][-6:-1]
                    history = np.array(history_fict )
                    feature = torch.zeros_like(data['inputs']['surrounding_agent_representation']['vehicles'][:,n,:,:])
                    feature[:,:,:2] = data['ground_truth']['traj'][:,-6:-1]
                    feature[:,:,2:] = data['inputs']['target_agent_representation'][:,-1,2:].repeat(1,5,1)
                    data['inputs']['surrounding_agent_representation']['vehicles'][:,n,:,:] = feature
                elif False: #ann['instance_token'] == veh_token:
                    history = history_fict
                    data['inputs']['surrounding_agent_representation']['vehicles'][:,n,:,:] = feature 
                elif self.example==2 and self.counterfactual and n==14:
                    # change bycicle trajectory
                    history = history - (history[-1]-history[0])*1.3
                    future[ann['instance_token']] = future[ann['instance_token']] - (history[-1]-history[0])*1.3
                    feature = data['inputs']['surrounding_agent_representation']['vehicles'][4,:,:]
                    feature[:,:2] = convert_global_coords_to_local(history, annotations[36]['translation'], annotations[36]['rotation'])  
                    data['inputs']['surrounding_agent_representation']['vehicles'][4,:,:] = feature
                    data['inputs']['surrounding_agent_representation']['adj_matrix'][0,4+1] = 1
                    data['inputs']['surrounding_agent_representation']['adj_matrix'][4+1,0] = 1    
                    data['inputs']['agent_node_masks']['vehicles'][:,4] = data['inputs']['agent_node_masks']['vehicles'][:,1]                
                ax2.plot(history[:, 0], history[:, 1], 'k--' )

                #Plot future
                if len(future[ann['instance_token']]) > 0:
                    future_plot = future[ann['instance_token']]  
                    ax2.plot(future_plot[:,0], future_plot[:,1], 'w--' )  
                    
                # Current Node Position
                node_circle_size=0.4
                circle_edge_width=0.5
                if ann['instance_token'] == i_t:
                    agent_translation = ann['translation']
                    agent_rotation = ann['rotation']
                    r_img = rotate(agent, quaternion_yaw(Quaternion(ann['rotation']))*180/math.pi,reshape=True)
                    oi = OffsetImage(r_img, zoom=0.015, zorder=500 )
                    veh_box = AnnotationBbox(oi, (history[-1, 0], history[-1, 1]), frameon=False)
                    veh_box.zorder = 800
                    ax2.add_artist(veh_box)                  
                elif ann['category_name'].split('.')[1] == 'motorcycle' or ann['category_name'].split('.')[1] == 'bicycle':
                    circle = plt.Circle((history[-1, 0],
                                history[-1, 1]),
                                node_circle_size,
                                facecolor='g',
                                edgecolor='g',
                                lw=circle_edge_width,
                                zorder=3)
                    ax2.add_artist(circle)
                elif ann['category_name'].split('.')[0] == 'vehicle': 
                    r_img = rotate(cars, quaternion_yaw(Quaternion(ann['rotation']))*180/math.pi,reshape=True)
                    oi = OffsetImage(r_img, zoom=0.015, zorder=5)
                    veh_box = AnnotationBbox(oi, (history[-1, 0], history[-1, 1]), frameon=False)
                    veh_box.zorder = 5
                    ax2.add_artist(veh_box)                 
                    # ax2.annotate(str(n), (history[-1, 0], history[-1, 1]), fontsize=10, zorder=10)
                elif ann['category_name'].split('.')[0] == 'human': 
                    circle = plt.Circle((history[-1, 0],
                                history[-1, 1]),
                                node_circle_size,
                                facecolor='c',
                                edgecolor='c',
                                lw=circle_edge_width,
                                zorder=3)
                    ax2.add_artist(circle)
                else: 
                    circle = plt.Circle((history[-1, 0],
                                history[-1, 1]),
                                node_circle_size,
                                facecolor='y',
                                edgecolor='y',
                                lw=circle_edge_width,
                                zorder=3, label='object')
                    ax2.add_artist(circle)
                
            # Include fictitious agent
            if self.counterfactual and self.example!=2:
                if self.example==5:
                    fict_idx = 5 
                else:  
                    fict_idx = 3
                history_fict = np.repeat(future[i_t][fict_idx:fict_idx+1],5,0)
                future_fict = np.repeat(future[i_t][fict_idx:fict_idx+1],12,0)
                ax2.plot(history_fict[:, 0], history_fict[:, 1], 'k--')
                ax2.plot(future_fict[:, 0], future_fict[:, 1], 'w--')
                r_img = rotate(cars, quaternion_yaw(Quaternion(ann['rotation']))*180/math.pi,reshape=True)
                rotation_fict = ann['rotation']
                oi = OffsetImage(r_img, zoom=0.015, zorder=5)
                veh_box = AnnotationBbox(oi, (history_fict[-1, 0], history_fict[-1, 1]), frameon=False)
                veh_box.zorder = 5
                ax2.add_artist(veh_box) 
                vehicle_mask = data['inputs']['surrounding_agent_representation']['vehicle_masks']     
                num_v = np.where(vehicle_mask[:,:,0]==0)[0].max()+1 if len(np.where(vehicle_mask[:,:,0]==0)[0])>0 else 0
                feature_fict = np.zeros_like(data['inputs']['surrounding_agent_representation']['vehicles'][num_v,:,:])
                feature_fict[:,:2] = np.tile(data['ground_truth']['traj'][fict_idx:fict_idx+1], (5,1))
                feature_fict[:,-1:] = np.tile(data['inputs']['target_agent_representation'][-1,-1],(5,1))
                data['inputs']['surrounding_agent_representation']['vehicles'][num_v,:,:] = feature_fict
                data['inputs']['surrounding_agent_representation']['vehicle_masks'][num_v,:,:] = np.zeros_like(data['inputs']['surrounding_agent_representation']['vehicle_masks'][num_v,:,:])
                data['inputs']['agent_node_masks']['vehicles'][:,num_v] = data['inputs']['agent_node_masks']['vehicles'][:,0]
                # Add fictitious agent to adjacency matrix  
                data['inputs']['surrounding_agent_representation']['adj_matrix'][0,num_v+1] = 1
                data['inputs']['surrounding_agent_representation']['adj_matrix'][num_v+1,0] = 1
                data['inputs']['surrounding_agent_representation']['adj_matrix'][num_v+1,num_v+1] = 1
                

            # Plot visited lanes and mask 
            if self.mask_lanes: 
                mask_out_lanes = []
                node_feats = data['inputs']['map_representation']['lane_node_feats'] 
                evf_gt = data['ground_truth']['evf_gt'] 
                snext = data['inputs']['map_representation']['s_next']  
                for node_id in data['inputs']['node_seq_gt'].astype(int)[data['inputs']['node_seq_gt'].astype(int)<len(node_feats)][:3]:
                    node_feat = node_feats[node_id]
                    next_lanes = snext[node_id][evf_gt[node_id]==1].astype(int)  # 
                    feat_len = np.sum(np.sum(np.absolute(node_feat), axis=1) != 0)
                    # global_node_coords = convert_local_coords_to_global(node_feat[:feat_len,:2], agent_translation, agent_rotation)
                    # Plot visited lanes
                    # ax2.scatter(global_node_coords[:, 0],global_node_coords[:, 1], s=40, color='r', alpha=0.8)
                    # Plot lanes visited in the future
                    for next_lane in next_lanes[:]:
                        if next_lane < len(node_feats):
                            feat_len = np.sum(np.sum(np.absolute(node_feats[next_lane]), axis=1) != 0)
                            global_node_coords = convert_local_coords_to_global(node_feats[next_lane][:feat_len,:2], agent_translation, agent_rotation)
                            ax2.scatter(global_node_coords[:, 0],global_node_coords[:, 1], s=40, color='r', alpha=0.8) 
                            data['inputs']['map_representation']['lane_node_masks'][next_lane,:,:] = np.ones_like(data['inputs']['map_representation']['lane_node_masks'][0,:,:]) 
                            data['inputs']['agent_node_masks']['vehicles'][next_lane] = np.ones_like(data['inputs']['agent_node_masks']['vehicles'][0]) 
                            data['inputs']['map_representation']['succ_adj_matrix'][:,next_lane] = np.zeros_like(data['inputs']['map_representation']['succ_adj_matrix'][:,0])
                            data['inputs']['map_representation']['succ_adj_matrix'][next_lane] = np.zeros_like(data['inputs']['map_representation']['succ_adj_matrix'][next_lane])
                            data['inputs']['map_representation']['prox_adj_matrix'][:,next_lane] = np.zeros_like(data['inputs']['map_representation']['prox_adj_matrix'][:,next_lane])
                            data['inputs']['map_representation']['prox_adj_matrix'][next_lane] = np.zeros_like(data['inputs']['map_representation']['prox_adj_matrix'][next_lane]) 
                            mask_out_lanes.append(next_lane)
                            data['inputs']['map_representation']['mask_out_lanes'] = mask_out_lanes

            # Predict             
            if 'scout' in self.encoder_type:
                data = self.collate_fn([data])
            data = u.send_to_device(u.convert_double_to_float(u.convert2tensors(data)))
            data['inputs']['att'] = True 
            predictions = self.model(data['inputs'])
            predictions['probs'][0], probs_ord_idcs = predictions['probs'].sort( dim=1, descending=True) 
            probs_ord_idcs = torch.flip(probs_ord_idcs[:self.num_modes_to_show], dims=(1,)) 
            predictions['probs'][0] = torch.flip(predictions['probs'][0][:self.num_modes_to_show], dims=(0,))
            predictions['traj'][0] = predictions['traj'][0][probs_ord_idcs[0]] 

            fig3, ax3 = plt.subplots(figsize=(12,12))
            #cmap_cool = plt.get_cmap('cool')
            #sm_cool = plt.cm.ScalarMappable(cmap=cmap_cool , norm=plt.Normalize(vmin=0, vmax=1)) 
            #cbar_cool = plt.colorbar(sm_cool)
            #cbar_cool.set_label('Probability of each cluster', rotation=270) 
            if self.show_predictions: 
                for n, traj in enumerate(predictions['traj'][0]):
                    """ ax[1].plot(traj[:, 0].detach().cpu().numpy(), traj[:, 1].detach().cpu().numpy(), lw=4,
                            color='r', alpha=0.8)
                    ax[1].scatter(traj[-1, 0].detach().cpu().numpy(), traj[-1, 1].detach().cpu().numpy(), 60,
                                color='r', alpha=0.8) """
                    global_traj = convert_local_coords_to_global(traj.detach().cpu().numpy(), agent_translation, agent_rotation)
                    if n == 0:
                        ax2.plot(global_traj[:self.tf, 0], global_traj[:self.tf, 1], color=cmap_cool(4*predictions['probs'][0][n].detach().cpu().numpy()), 
                                        lw=max(1,7*predictions['probs'][0][n]), linestyle = '--', alpha=0.8,label='Predicted Trajectory')
                    else: 
                        ax2.plot(global_traj[:self.tf, 0], global_traj[:self.tf, 1], color=cmap_cool(4*predictions['probs'][0][n].detach().cpu().numpy()), 
                                        lw=max(1,7*predictions['probs'][0][n]), linestyle = '--', alpha=0.8) 
                    if False: #'scout' in self.encoder_type and  n==0:
                        # Remove vehicles that are masked out
                        node_v_feats = data['inputs']['surrounding_agent_representation']['vehicles'][0][data['inputs']['surrounding_agent_representation']['vehicle_masks'][0,:,:,0].sum(-1) < 5]
                        node_p_feats = data['inputs']['surrounding_agent_representation']['pedestrians'][0][data['inputs']['surrounding_agent_representation']['pedestrian_masks'][0,:,:,0].sum(-1) < 5]
                        lane_feats = data['inputs']['map_representation']['lane_node_feats'][0]

                        # Retrieve object-level edge attention -> att[3:]. att[:3] is type-level attention.
                        att = [att.detach().cpu().numpy()  for attention in predictions['att'] for att in attention.values()]
                        graph = data['inputs']['lanes_graphs'].cpu()
                        vehicles_feats=np.concatenate((data['inputs']['target_agent_representation'][:,-1:,:2].detach().cpu().numpy(), node_v_feats[:,-1:,:2].detach().cpu().numpy()))
                        pedestrians_feats = node_p_feats[:,-1:,:2].detach().cpu().numpy()

                        self.visualize_graph(fig3,ax3,lane_feats.detach().cpu().numpy(), data['inputs']['map_representation']['s_next'][0].detach().cpu().numpy(), 
                            data['inputs']['map_representation']['edge_type'][0].detach().cpu().numpy(), data['ground_truth']['evf_gt'][0].detach().cpu().numpy(), 
                            data['inputs']['node_seq_gt'][0].detach().cpu().numpy(), traj.detach().cpu().numpy(), predictions['pi'][0].detach().cpu().numpy(), 
                            cmap_cool, vehicles_feats, graph, att)

                        for i, vehicle in enumerate(vehicles_feats):
                            vehicles_feats[i] = convert_local_coords_to_global(vehicle, agent_translation, agent_rotation)
                        for i, pedestrian in enumerate(pedestrians_feats):
                            pedestrians_feats[i] = convert_local_coords_to_global(pedestrian, agent_translation, agent_rotation)
                        v2v_attn = att[-1]
                        v2l_attn = att[-2]  
                        p2v_attn = att[-3] 
                        # Plot interactions with other agents for the focal vehicle
                        if len(v2v_attn.shape) > 0:
                            for idx, v in enumerate(graph['v_interact_v'].edges()[0]):
                                # if v > graph['v_interact_v'].edges()[1][idx]:
                                if graph['v_interact_v'].edges()[0][idx] == 0:
                                    color = 'w'
                                    # Compute attention taking into account object-level and type-level attention
                                    attn = v2v_attn[idx]*att[2][-1][graph['v_interact_v'].edges()[1][idx]]*2.5 
                                    w = attn
                                    alpha = 1
                                    ax2.arrow(vehicles_feats[v][-1,0], vehicles_feats[v][-1,1], vehicles_feats[graph['v_interact_v'].edges()[1][idx]][-1,0]-vehicles_feats[v][-1,0], 
                                        vehicles_feats[graph['v_interact_v'].edges()[1][idx]][-1,1]-vehicles_feats[v][-1,1], color=color, head_width=0.1, 
                                        length_includes_head=True,  width=w, alpha=alpha) 
                        # Plot interactions with pedestrians for the focal vehicle
                        if len(p2v_attn.shape) > 0:
                            for idx, p in enumerate(graph['p_interact_v'].edges()[0]):
                                # Visualize only those pedestrians who interact with focal agent
                                if graph['p_interact_v'].edges()[1][idx] == 0:
                                    color = 'r'
                                    # Compute attention taking into account object-level and type-level attention
                                    attn = p2v_attn[idx]   #att[2][-2][graph['p_interact_v'].edges()[0][idx]] 
                                    w = attn
                                    alpha = 1
                                    ax2.arrow(vehicles_feats[0][-1,0], vehicles_feats[0][-1,1], pedestrians_feats[graph['p_interact_v'].edges()[0][idx]][-1,0]-vehicles_feats[0][-1,0], 
                                        pedestrians_feats[graph['p_interact_v'].edges()[0][idx]][-1,1]-vehicles_feats[0][-1,1], color=color, head_width=0.1, 
                                        length_includes_head=True,  width=w, alpha=alpha)

                        
            
            if self.legend:
                legend=ax2.legend(frameon=True, loc='upper right', facecolor='lightsteelblue', edgecolor='black', fontsize=10)
                handles, labels = ax2.get_legend_handles_labels() 
                idx = labels.index("lane")
                handles.pop(idx)
                labels.pop(idx)
                idx=labels.index("road_block")
                handles.pop(idx)
                labels.pop(idx)
                labels.append("2s past trajectory")
                handles.append(ax2.plot(history[:, 0], history[:, 1], 'k--' )[0]) 
                labels.append("6s future ground truth trajectory")
                handles.append(ax2.plot(future_plot[:, 0], future_plot[:, 1], 'w--' )[0]) 
                handles.append(Patch(facecolor='red', edgecolor='r'))
                labels.append("Autonomous Vehicle")
                handles.append(Patch(facecolor='darkblue', edgecolor='b'))
                labels.append("Focal vehicle")
                handles.append(Patch(facecolor='lightblue', edgecolor='c')) 
                labels.append("Surrounding vehicles")
                handles.append(Line2D([], [], color="c", marker='o', markersize=6,  markerfacecolor="c", markeredgecolor="c", linestyle='None'))
                labels.append("Pedestrians")
                handles.append(Line2D([], [], color="g", marker='o', markersize=6,  markerfacecolor="g", markeredgecolor="g", linestyle='None'))
                labels.append("Bycicles")
                handles.append(Line2D([], [], color="y", marker='o', markersize=6,  markerfacecolor="y", markeredgecolor="y", linestyle='None'))
                labels.append("Objects")
                legend._legend_box = None
                legend._init_legend_box(handles, labels)
                legend._set_loc(legend._loc)
                legend.set_title(legend.get_title().get_text())
            

            fig2.canvas.draw()
            image_from_plot = np.frombuffer(fig2.canvas.tostring_rgb(), dtype=np.uint8) 
            image_from_plot = image_from_plot.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
            imgs_fancy.append(image_from_plot)   
            
            plt.close(fig2) 

            fig3.canvas.draw( )
            image_from_plot = np.frombuffer(fig3.canvas.tostring_rgb(), dtype=np.uint8) 
            image_from_plot = image_from_plot.reshape(fig3.canvas.get_width_height()[::-1] + (3,))
            graph_img.append(image_from_plot)
            plt.close(fig3) 

        return imgs_fancy,graph_img, scene_name
