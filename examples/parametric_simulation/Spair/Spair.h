//
// Created by rsltgy on 11/12/2020.
//

#ifndef LIBGRAPE_LITE_SPAIR_H
#define LIBGRAPE_LITE_SPAIR_H

#include <grape/grape.h>
#include "../ParaMatch/ParaMatch.h"
#include "Spair_context.h"
#include "grape/utils/vertex_array.h"

using namespace std;


namespace grape{

    template <typename FRAG_T>
    class Spair : public ParallelAppBase<FRAG_T, SpairContext<FRAG_T>>,
                  public ParallelEngine {


        // specialize the templated worker.
    INSTALL_PARALLEL_WORKER(Spair<FRAG_T>, SpairContext<FRAG_T>, FRAG_T)
        using vertex_t = typename fragment_t::vertex_t;

        //static constexpr MessageStrategy message_strategy =
        //        MessageStrategy::kAlongEdgeToOuterVertex;

        static constexpr LoadStrategy load_strategy = LoadStrategy::kOnlyOut;


        void PEval(const fragment_t& frag, context_t& ctx,
                   message_manager_t& messages) {

            auto &rev = ctx.rev;
            auto &GD = ctx.GD;
            auto &u = ctx.u;
            auto &v = ctx.v;
            auto &cache = ctx.cache;
            auto &ecache_u = ctx.ecache_u;
            auto &ecache_v = ctx.ecache_v;
            auto &g_paths = ctx.g_paths;
            vector<vector<int>> &g_descendants = ctx.g_descendants;
            auto &word_embeddings = ctx.word_embeddings;
            double &sigma = ctx.sigma;
            double &delta = ctx.delta;
            ParaMatch<FRAG_T> p;
            messages.InitChannels(thread_num(), 2 * 1023 * 64, 2 * 1024 * 64);
            auto& channel_0 = messages.Channels()[0];

            // We use this part if the sim between u and v is below than sigma
            bool pass_message = false;
            vector<double> u_word_vector;
            Reader::calculate_word_vector(word_embeddings,GD.nodes()[u],u_word_vector);
            vertex_t frag_vert;
            if(!frag.GetInnerVertex(v,frag_vert))
                pass_message = true;

            if (!pass_message){
                string v_data;
                v_data = frag.GetData((vertex_t)frag_vert);
                vector<double> v_word_vector;
                boost::trim_right(v_data);
                Reader::calculate_word_vector(word_embeddings,v_data,v_word_vector);
                double score = p.cosine_similarity(u_word_vector,v_word_vector);
                if (score < sigma){
                    pass_message = true;
                }
            }

            if(!pass_message){
                //p.match(ctx.GD,frag,ctx.g_paths,ctx.g_descendants,ctx.word_embeddings,u,v,sigma,delta);
                bool result_of_spair = p.match_pair(GD,frag,g_paths,g_descendants,u,v,sigma,delta,cache,word_embeddings,ecache_u,ecache_v,rev);
                /*for (auto &ca : ctx.cache) {
                    cout << "Fragment " << frag.fid() << " (" << ca.first.first
                         << "," << ca.first.second << ") -> " << ca.second.first << " " << ca.second.second.size()
                         << endl;
                }*/
                if(!result_of_spair){ // if spair of u and v is false, then do further calculation otherwise do nothing.
                    vertex_t frag_vert;
                    vertex_t o_v;
                    if(frag.GetInnerVertex(v,frag_vert)){                // if vertex v is in this fragment then sent message to other
                        auto outer_vertices = frag.OuterVertices();
                        map<unsigned int,std::pair<unsigned int,vector<std::pair<int,int>>>> msg;

                        for (vertex_t o_v : outer_vertices ) {
                            unsigned int fid = frag.GetFragId(o_v);
                            msg[fid].first = frag.fid();
                            //for(int n = 0 ; n < GD.number_of_nodes(); n++)
                            for(int n = 0 ; n < 1; n++) // instead of all GD, just send a key to check all vertices in GD
                                msg[fid].second.push_back(std::make_pair(-1,frag.Gid2Oid(frag.Vertex2Gid(o_v))));
                            //cout << " fid " << fid << " " << frag.Gid2Oid(frag.Vertex2Gid(o_v)) << endl;
                        }
                        for (auto m : msg ) {
                            //channel_0.SyncStateOnOuterVertex<fragment_t,vector<std::pair<int,int>>>(frag,o_v,msg);
                            channel_0.SendToFragment(m.first,m.second);
                        }

                        auto witness_vertices =  cache[make_pair(ctx.u,ctx.v)].second;
                        double &sum = ctx.sum;
                        ParaMatch<FRAG_T> p;
                        for(auto wit : witness_vertices){
                            double local_sum = p.calculate_path_similarity(GD,g_paths,ctx.word_embeddings, ctx.u, wit.first, v, wit.second);
                            sum += local_sum;
                            //cout << " sum of " << ctx.u << " " << wit.first << " " << v << " " << wit.second  << " is " << local_sum << " total " << sum << endl;
                        }
                    }
                }else{
                    ctx.result = true;
                }
            }else{
                cout << "frag " << frag.fid() << " does not have vertex " << v << endl;
            }
        }

        void IncEval(const fragment_t& frag, context_t& ctx,
                     message_manager_t& messages) {

            auto &rev = ctx.rev;
            auto &GD = ctx.GD;
            auto &v = ctx.v;
            auto &cache = ctx.cache;
            auto &ecache_u = ctx.ecache_u;
            auto &ecache_v = ctx.ecache_v;
            auto &g_paths = ctx.g_paths;
            auto &g_descendants = ctx.g_descendants;
            auto &word_embeddings = ctx.word_embeddings;
            double &sigma = ctx.sigma;
            double &delta = ctx.delta;
            auto& channel_0 = messages.Channels()[0];
            double &sum = ctx.sum;

            vector<std::pair<unsigned int,vector<std::pair<int,int>>>>  messages_received;
            messages.ParallelProcess<std::pair<unsigned int,vector<std::pair<int,int>>>>(
                    1, [&frag,&messages_received](int tid, std::pair<unsigned int,vector<std::pair<int,int>>> msg) {
                        messages_received.push_back(msg);
                    });
            ParaMatch<FRAG_T> p;
            vertex_t source_vert;
            for(auto message : messages_received){
                if(!message.second.empty()) {
                    bool fragment_has_everything = false;
                    for (auto m : message.second) {
                        auto pair_received = std::make_pair(m.first, m.second);
                        if(m.first == -1){
                            // This call comes from Peval, execute all GD.
                            for(int n = 0 ; n < GD.number_of_nodes(); n++){
                                p.match_pair(GD, frag, g_paths, g_descendants, n, m.second, sigma, delta, cache,
                                             word_embeddings, ecache_u, ecache_v,rev);
                                if(cache[std::make_pair(n, m.second)].first){
                                    fragment_has_everything = true;
                                    break;
                                }
                            }
                        }else if(cache.find(pair_received) == cache.end()){
                            p.match_pair(GD, frag, g_paths, g_descendants, m.first, m.second, sigma, delta, cache,
                                         word_embeddings, ecache_u, ecache_v,rev);
                            if(cache[pair_received].first){
                              fragment_has_everything = true;
                            }
                        }
                    }
                    if(fragment_has_everything){
                        map<unsigned int,std::pair<unsigned int,vector<std::pair<int,int>>>> msg;
                        for (auto &ca : cache) {
                            cout << "Fragment " << frag.fid() << " (" << ca.first.first
                                 << "," << ca.first.second << ") -> " << ca.second.first << " " << ca.second.second.size()
                                 << endl;
                            if (ca.second.first) {
                                vertex_t frag_vert;
                                frag.GetVertex(ca.first.first, frag_vert);
                                msg[message.first].first = frag.fid();
                                msg[message.first].second.push_back(ca.first);
                            }
                        }

                        if(msg.empty()){   // This is for cycles
                            for (vertex_t o_v : frag.OuterVertices()) {
                                unsigned int fid = frag.GetFragId(o_v);
                                msg[fid].first = frag.fid();
                                for(int n = 0 ; n < 1; n++)
                                    msg[fid].second.push_back(std::make_pair(-1,frag.Gid2Oid(frag.Vertex2Gid(o_v))));
                            }
                        }
                        for (auto m : msg ) {
                            cout << " frag " << frag.fid() << " is sending a message to frag" << m.first << endl;
                            channel_0.SendToFragment(m.first,m.second);
                        }
                    }else{

                        for (auto y : message.second) {
                            cout << " frag " << frag.fid() << " got message from " << message.first << " " << y.first
                                 << " " << y.second << endl;
                            double local_sum = p.calculate_path_similarity(GD, g_paths, ctx.word_embeddings, ctx.u,
                                                                           y.first, v, y.second);
                            sum += local_sum;
                            cout << " sum of " << ctx.u << " " << y.first << " " << v << " " << y.second << " is "
                                 << local_sum << " total " << sum << endl;
                            if (sum >= ctx.delta) {
                                cout << sum << " Vertex u " << ctx.u << " and " << ctx.v << " is a match " << endl;
                                ctx.result = true;
                                break;
                            }
                        }


                    }
                }
                /*cout << "here "<< endl;
                for (auto &ca : cache) {
                    cout << "Fragment " << frag.fid() << " (" << ca.first.first
                         << "," << ca.first.second << ") -> " << ca.second.first << " " << ca.second.second.size()
                         << endl;
                }*/
            }
        }

    };



}


#endif  // LIBGRAPE_LITE_SPAIR_H
