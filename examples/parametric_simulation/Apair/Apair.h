//
// Created by rsltgy on 20/12/2020.
//

#ifndef LIBGRAPE_LITE_APAIR_H
#define LIBGRAPE_LITE_APAIR_H


#include <grape/grape.h>
#include "../ParaMatch/ParaMatch.h"
#include "Apair_context.h"
#include "grape/utils/vertex_array.h"
#include "../KDTree/KDTree.h"

using namespace std;


namespace grape{

    template <typename FRAG_T>
    class Apair : public ParallelAppBase<FRAG_T, ApairContext<FRAG_T>>,
                  public ParallelEngine {



// specialize the templated worker.
    INSTALL_PARALLEL_WORKER(Apair<FRAG_T>, ApairContext<FRAG_T>, FRAG_T)
        using vertex_t = typename fragment_t::vertex_t;


        static constexpr MessageStrategy message_strategy = MessageStrategy::kAlongIncomingEdgeToOuterVertex;
        static constexpr LoadStrategy load_strategy = LoadStrategy::kOnlyIn;


        void PEval(const fragment_t& frag, context_t& ctx,
                   message_manager_t& messages) {

            auto &rev = ctx.rev;
            auto &match_set = ctx.match_set;
            auto &C = ctx.C;
            auto &GD = ctx.GD;
            auto &cache = ctx.cache;
            auto &ecache_u = ctx.ecache_u;
            auto &ecache_v = ctx.ecache_v;
            auto &g_paths = ctx.g_paths;
            auto &g_descendants = ctx.g_descendants;
            auto &word_embeddings = ctx.word_embeddings;
            auto &message_cache = ctx.message_cache;
            auto &sigma = ctx.sigma;
            auto &delta = ctx.delta;
            messages.InitChannels(thread_num(), 2 * 1023 * 64, 2 * 1024 * 64);
            auto& channel_0 = messages.Channels()[0];
            cout << "Candidates generation started " << endl;
            pointVec points;
            auto inner_vert = frag.InnerVertices();
            for (auto v : inner_vert ) {
                string v_str = frag.GetData(v);
                //auto oid = frag.Gid2Oid(frag.Vertex2Gid(v));
                //cout << frag.fid() << " " << oid  << " " << v_str << endl;
                vector<double> v_g_word_vector;
                boost::trim_right(v_str);
                Reader::calculate_word_vector(ctx.word_embeddings,v_str,v_g_word_vector);
                if(!v_g_word_vector.empty()){
                    auto oid = frag.Gid2Oid(frag.Vertex2Gid(v));
                    point_t t = make_pair(oid,v_g_word_vector);
                    points.push_back(t);
                    //cout << v_str <<  " " << oid << endl;
                }
            }

            KDTree tree(points);

            for(int u_t = 0; u_t < GD.number_of_nodes(); u_t++){
                //cout << frag.fid() << " " <<  u_t << endl;
                if(!GD.nodes()[u_t].empty()){
                    vector<double> u_t_word_vector;
                    Reader::calculate_word_vector(word_embeddings,GD.nodes()[u_t],u_t_word_vector);
                    if(!u_t_word_vector.empty()) {
                        point_t t = make_pair(u_t,u_t_word_vector);
                        auto NNs = tree.neighborhood_points(t, sigma);
                        for(auto returned_match : NNs){
                            //vertex_t v(returned_match.first);
                            //auto es = frag.GetOutgoingAdjList(v);
                            C.push_back(make_pair(u_t,make_pair(returned_match.first,u_t)));
                            //cout << frag.fid() << " " <<  u_t << " " << returned_match.first << endl;
                        }
                    }
                }
            }
            cout << "Candidates generated " << C.size() << endl;
            points.clear();
            points.shrink_to_fit();

            sort(C.begin(),C.end(),[](const pair<int,pair<int,int>> &a, const pair<int,pair<int,int>> b){
                return a.second.second < b.second.second;
            });

            bool match = false;
            for(auto v : C){
                //cout << frag.fid() << " ) " << " " << v.first << " " << v.second.first << endl;
                auto it = cache.find(make_pair(v.first,v.second.first));
                if(it != ctx.cache.end() && cache[make_pair(v.first,v.second.first)].first  ){
                    match_set[v.second.first].push_back(v.first);
                    cout << " here " << v.first<< " " << v.second.first << endl;
                }
                else{
                    ParaMatch<FRAG_T> p;
                    match = p.match_pair(GD,frag,g_paths,g_descendants,v.first,v.second.first,sigma,delta,cache,word_embeddings,ecache_u,ecache_v,rev);
                    if(match)
                        match_set[v.second.first].push_back(v.first);
                }
            }
            cout << " PEval Done" << endl;

            /*auto outer_vertices = frag.OuterVertices();
            for (vertex_t o_v : outer_vertices ) {
                auto oid = frag.Gid2Oid(frag.Vertex2Gid(o_v));
                cout << frag.fid() << " " << oid  <<  endl;
            }*/

            for (auto &ca : cache) {
                //cout << "Fragment " << frag.fid() << " (" << ca.first.first
                //     << "," << ca.first.second << ") -> " << ca.second.first << " " << ca.second.second.size() << endl;
                message_cache[ca.first.second].push_back(ca.first.first);
            }


            auto inner_vertices = frag.InnerVertices();
            for(auto i_v : inner_vertices){
                auto oid = frag.Gid2Oid(frag.Vertex2Gid(i_v));
                //cout << frag.fid() << " " << oid  <<  endl;
                if(frag.IsIncomingBorderVertex(i_v)){
                    //cout << " this is iborder vertex " <<  endl;
                    std::pair<int,vector<int>> msg;
                    if(match_set.find(oid) != match_set.end()){
                       channel_0.SendMsgThroughIEdges(frag,i_v,std::make_pair(oid,match_set[oid]));
                    }
                }
            }

        }


        void IncEval(const fragment_t& frag, context_t& ctx,
                     message_manager_t& messages) {
            auto &rev = ctx.rev;
            auto &GD = ctx.GD;
            auto &cache = ctx.cache;
            auto &ecache_u = ctx.ecache_u;
            auto &ecache_v = ctx.ecache_v;
            auto &g_paths = ctx.g_paths;
            auto &g_descendants = ctx.g_descendants;
            auto &word_embeddings = ctx.word_embeddings;
            double &sigma = ctx.sigma;
            double &delta = ctx.delta;
            auto &match_set = ctx.match_set;
            auto &message_cache = ctx.message_cache;
            auto& channel_0 = messages.Channels()[0];

            unordered_map<vertex_t, vector<std::pair<int,vector<int>>>> messages_received;
            messages.ParallelProcess<fragment_t, std::pair<int,vector<int>>>(
                    1, frag, [&messages_received](int tid, vertex_t v, std::pair<int,vector<int>> msg) {
                        messages_received[v].push_back(msg);
                    });

            for(auto message : messages_received){
                auto v = frag.Gid2Oid(frag.Vertex2Gid(message.first));
                cout << "Node "  << v << endl;
                std::pair<int,vector<int>> msg;
                msg.first = v;
                vector<int> v_s = message_cache[v];
                for(int u : v_s){
                    pair<bool, vector<pair<int, int>>> match_result_and_witnesses = cache[std::make_pair(u,v)];
                    if(!match_result_and_witnesses.first){
                        auto witness_vertices =  match_result_and_witnesses.second;
                        double &sum = ctx.sum;
                        ParaMatch<FRAG_T> p;
                        for(auto wit : witness_vertices){
                            double local_sum = p.calculate_path_similarity(GD,g_paths,ctx.word_embeddings, u, wit.first, v, wit.second);
                            sum += local_sum;
                            cout << " sum of " << u << " " << wit.first << " " << v << " " << wit.second  << " is " << local_sum << " total " << sum << endl;
                        }

                        for(auto  v_prime_and_all_u_primes : message.second ){
                            int v_prime = v_prime_and_all_u_primes.first;
                            for(auto u_prime : v_prime_and_all_u_primes.second){
                                double local_sum = p.calculate_path_similarity(GD,g_paths,ctx.word_embeddings, u, u_prime , v, v_prime);
                                sum += local_sum;
                                cout << " sum of " << u << " " << u_prime << " " << v << " " << v_prime << " is " << local_sum << " total " << sum << endl;
                                if (sum >= ctx.delta)  {
                                    cout << sum << " Vertex u " << u << " and " << v << " is a match " << endl;
                                    match_set[u].push_back(v);
                                    msg.second.push_back(u);
                                    auto u_v = make_pair(u, v);
                                    cache[u_v].first = false;

                                    for(const auto &u_p_v_p: rev[u_v]) {
                                        cache.erase(u_p_v_p);
                                    }

                                    for(const auto &u_p_v_p: rev[u_v]) {
                                        p.match_pair(GD, frag, g_paths,g_descendants,u_p_v_p.first, u_p_v_p.second, sigma, delta, cache, word_embeddings, ecache_u, ecache_v, rev);
                                    }
                                    rev[u_v].clear();
                                    break;
                                }
                            }
                            if (sum >= ctx.delta)
                                break;
                        }

                    }
                }


                if(!msg.second.empty()){
                    auto inner_vertices = frag.InnerVertices();
                    for(auto i_v : inner_vertices){
                        auto oid = frag.Gid2Oid(frag.Vertex2Gid(i_v));
                        if(frag.IsIncomingBorderVertex(i_v) && v == oid){
                            channel_0.SendMsgThroughIEdges(frag,i_v,std::make_pair(oid,match_set[oid]));
                        }
                    }

                }


            }

        }
    };



}


#endif  // LIBGRAPE_LITE_APAIR_H
