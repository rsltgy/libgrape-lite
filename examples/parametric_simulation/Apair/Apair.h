

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



    INSTALL_PARALLEL_WORKER(Apair<FRAG_T>, ApairContext<FRAG_T>, FRAG_T)
        using vertex_t = typename fragment_t::vertex_t;


        static constexpr MessageStrategy message_strategy = MessageStrategy::kSyncOnOuterVertex;
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
            auto &message_address = ctx.message_address;
            auto &sigma = ctx.sigma;
            auto &delta = ctx.delta;
            messages.InitChannels(thread_num(), 2 * 1023 * 64, 2 * 1024 * 64);
            auto& channel_0 = messages.Channels()[0];

            timer_next("Candidate Generation");

            auto inner_vert = frag.InnerVertices();
            {
                pointVec points;

                points.resize(inner_vert.size());

                ForEach(inner_vert, [&](int tid, vertex_t v) {
                    string v_str = frag.GetData(v);
                    vector<double> v_g_word_vector;
                    boost::trim_right(v_str);
                    Reader::calculate_word_vector(ctx.word_embeddings, v_str,
                                                  v_g_word_vector);
                    if (!v_g_word_vector.empty()) {
                        auto oid = frag.GetId(v);
                        points[v.GetValue()] = std::make_pair(oid, v_g_word_vector);
                    }
                });

                KDTree tree(points);

                for(int u_t = 0; u_t < GD.number_of_nodes(); u_t++){
                    if(!GD.nodes()[u_t].empty()){
                        vector<double> u_t_word_vector;
                        Reader::calculate_word_vector(word_embeddings,GD.nodes()[u_t],u_t_word_vector);
                        if(!u_t_word_vector.empty()) {
                            point_t t = make_pair(u_t,u_t_word_vector);
                            auto NNs = tree.neighborhood_points(t, sigma);
                            for(auto returned_match : NNs){
                                C.emplace_back(u_t, make_pair(returned_match.first, u_t));
                            }
                        }
                    }
                }
            }

            timer_next("Parametric Simulation from Candidates");
            sort(C.begin(),C.end(),[](const pair<int,pair<int,int>> &a, const pair<int,pair<int,int>> b){
                return a.second.second < b.second.second;
            });

            bool match = false;
            for(auto& v : C){
                auto pair = make_pair(v.first,v.second.first);
                auto it = cache.find(pair);
                if(it != ctx.cache.end() && cache[pair].first  ){
                    match_set[v.second.first].insert(v.first);
                }
                else{
                    ParaMatch<FRAG_T> p;
                    match = p.match_pair(GD,frag,g_paths,g_descendants,v.first,v.second.first,sigma,delta,cache,word_embeddings,ecache_u,ecache_v,rev);
                    if(match)
                        match_set[v.second.first].insert(v.first);
                }
            }
            for (auto &ca : cache) {
                message_cache[ca.first.second].push_back(ca.first.first);
            }
            auto inner_vertices = frag.InnerVertices();
            auto outer_vertices = frag.OuterVertices();

            timer_next("Message Address Finding");

            ForEach(outer_vertices, [&](int tid, vertex_t o_v) {
                auto o_v_oid = frag.GetId(o_v);
                for(auto &desc : g_descendants[o_v_oid]){
                    vertex_t frag_vert;
                    if(frag.GetVertex(desc,frag_vert)){
                        if(match_set.find(desc) != match_set.end()) {
                            messages.Channels()[tid].SyncStateOnOuterVertex(frag, o_v, make_pair(desc,match_set[desc]));
                        }
                    }
                }
            });
            timer_next("PEval");
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
            auto &message_address = ctx.message_address;
            auto& channel_0 = messages.Channels()[0];

            timer_next("IncEval");
            unordered_map<vertex_t, vector<pair<int,set<int>>>> messages_received;
            messages.ParallelProcess<fragment_t, pair<int,set<int>>>(
                    1, frag, [&ctx,&messages_received](int tid, vertex_t v, pair<int,set<int>> msg) {
                        messages_received[v].push_back(msg);
                    });

            for(auto &message : messages_received){
                auto message_come_from = frag.GetId(message.first);
                std::pair<int,vector<int>> msg;
                unsigned v = message_come_from;
                vector<int> u_s = message_cache[v];
                for(int u : u_s){
                    auto u_v = make_pair(u, v);
                    pair<bool, vector<pair<int, int>>> match_result_and_witnesses = cache[u_v];
                    if(!match_result_and_witnesses.first){
                        auto witness_vertices =  match_result_and_witnesses.second;
                        double sum = 0;
                        ParaMatch<FRAG_T> p;
                        for(auto& wit : witness_vertices){
                            double local_sum = p.calculate_path_similarity(GD,g_paths,ctx.word_embeddings, u, wit.first, v, wit.second);
                            sum += local_sum;
                        }

                        for(auto & v_prime_and_all_u_primes : message.second ){
                            unsigned int v_prime = v_prime_and_all_u_primes.first;
                            for(auto u_prime : v_prime_and_all_u_primes.second){
                                double local_sum = p.calculate_path_similarity(GD,g_paths,ctx.word_embeddings, u, u_prime , v, v_prime);
                                sum += local_sum;
                                if (sum >= ctx.delta)  {
                                    match_set[u].insert(v);
                                    msg.second.push_back(u);
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
                    auto outer_vertices = frag.OuterVertices();
                    ForEach(outer_vertices, [&](int tid, vertex_t o_v) {
                        auto o_v_oid = frag.GetId(o_v);
                        for(auto &desc : g_descendants[o_v_oid]){
                            vertex_t frag_vert;
                            if(frag.GetVertex(desc,frag_vert)){
                                if(match_set.find(desc) != match_set.end()) {
                                    messages.Channels()[tid].SyncStateOnOuterVertex(frag, o_v, make_pair(desc,match_set[desc]));
                                }
                            }
                        }
                    });
                }
            }

        }
    };



}


#endif  // LIBGRAPE_LITE_APAIR_H