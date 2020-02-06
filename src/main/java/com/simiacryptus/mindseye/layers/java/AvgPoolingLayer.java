/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.google.common.cache.CacheLoader;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class AvgPoolingLayer extends LayerBase {

  //  public static final LoadingCache<AvgPoolingLayer.IndexMapKey, RefMap<Coordinate, RefList<int[]>>> indexMapCache = CacheBuilder
//      .newBuilder().build(new LayerCacheLoader());
  public static final ConcurrentHashMap<IndexMapKey, RefMap<Coordinate, RefList<int[]>>> indexMapCache = new ConcurrentHashMap<>();

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(AvgPoolingLayer.class);
  private int[] kernelDims;

  protected AvgPoolingLayer() {
    super();
  }

  public AvgPoolingLayer(@Nonnull final int... kernelDims) {

    this.kernelDims = RefArrays.copyOf(kernelDims, kernelDims.length);
  }

  protected AvgPoolingLayer(@Nonnull final JsonObject id, @Nonnull final int... kernelDims) {
    super(id);
    this.kernelDims = RefArrays.copyOf(kernelDims, kernelDims.length);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static AvgPoolingLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new AvgPoolingLayer(json, JsonUtil.getIntArray(json.getAsJsonArray("heapCopy")));
  }

  @RefIgnore
  private static synchronized RefMap<Coordinate, RefList<int[]>> getCoordMap(final int[] kernelDims,
                                                                             final int[] outDims) {
    RefMap<Coordinate, RefList<int[]>> coordinateRefListRefMap;
    synchronized (indexMapCache) {
      coordinateRefListRefMap = indexMapCache.computeIfAbsent(new IndexMapKey(kernelDims, outDims), AvgPoolingLayer::getIndexMap);
    }
    return coordinateRefListRefMap.addRef();
//    try {
//    synchronized (AvgPoolingLayer.indexMapCache) {
//      coordinateRefListRefMap = AvgPoolingLayer.indexMapCache.get(new IndexMapKey(kernelDims, outDims));
//    }
//    } catch (@Nonnull final ExecutionException e) {
//      throw new RuntimeException(e);
//    }
  }

  @Nonnull
  @SuppressWarnings("unchecked")
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final int kernelSize = Tensor.length(kernelDims);
    final TensorList data = inObj[0].getData();
    @Nonnull final int[] inputDims = data.getDimensions();
    final int[] newDims = RefIntStream.range(0, inputDims.length).map(i -> {
      assert 0 == inputDims[i] % kernelDims[i] : inputDims[i] + ":" + kernelDims[i];
      return inputDims[i] / kernelDims[i];
    }).toArray();
    final RefMap<Coordinate, RefList<int[]>> coordMap = getCoordMap(kernelDims, newDims);
    final Tensor[] outputValues = RefIntStream.range(0, data.length())
        .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) dataIndex -> {
          @Nullable final Tensor input = data.get(dataIndex);
          @Nonnull final Tensor output = new Tensor(newDims);
          coordMap.forEach((k, v) -> {
            double sum = v.stream()
                .mapToDouble(RefUtil.wrapInterface(input::get, input.addRef()))
                .sum();
            v.freeRef();
            if (Double.isFinite(sum)) {
              output.add(k, sum / kernelSize);
            }
          });
          input.freeRef();
          return output;
        }, data, coordMap.addRef()))
        .toArray(Tensor[]::new);
    try {
      Result.Accumulator accumulator = new Result.Accumulator() {
        {
          RefUtil.addRefs(inObj);
          coordMap.addRef();
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
          if (inObj[0].isAlive()) {
            @Nonnull
            TensorArray tensorArray = new TensorArray(RefIntStream.range(0, delta.length())
                .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) dataIndex -> {
                      @Nullable
                      Tensor tensor = delta.get(dataIndex);
                      @Nonnull final Tensor backSignal = new Tensor(inputDims);
                      coordMap.forEach((k, v) -> {
                        final double outputValue = tensor.get(k);
                        v.forEach(inputCoord -> backSignal.add(inputCoord, outputValue / kernelSize));
                        v.freeRef();
                      });
                      tensor.freeRef();
                      return backSignal;
                    }, delta)
                ).toArray(Tensor[]::new));
            inObj[0].accumulate(buffer == null ? null : buffer.addRef(), tensorArray);
          } else {
            delta.freeRef();
          }
          if (null != buffer)
            buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          super._free();
          RefUtil.freeRef(inObj);
          coordMap.freeRef();
        }
      };
      coordMap.freeRef();
      return new Result(new TensorArray(outputValues), accumulator) {
        {
          RefUtil.addRefs(inObj);
        }

        @Override
        public boolean isAlive() {
          return inObj[0].isAlive();
        }

        public void _free() {
          RefUtil.freeRef(inObj);
          super._free();
        }
      };
    } finally {
      RefUtil.freeRef(inObj);
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("heapCopy", JsonUtil.getJson(kernelDims));
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  AvgPoolingLayer addRef() {
    return (AvgPoolingLayer) super.addRef();
  }

  public static final class IndexMapKey {
    final int[] kernel;
    final int[] output;

    public IndexMapKey(final int[] kernel, final int[] output) {
      super();
      this.kernel = kernel;
      this.output = output;
    }

    public IndexMapKey(@Nonnull final Tensor kernel, @Nullable final Tensor input, @Nonnull final Tensor output) {
      super();
      if (null != input)
        input.freeRef();
      this.kernel = kernel.getDimensions();
      kernel.freeRef();
      this.output = output.getDimensions();
      output.freeRef();
    }

    @Override
    public boolean equals(@Nullable final Object obj) {
      if (this == obj) {
        return true;
      }
      if (obj == null) {
        return false;
      }
      if (getClass() != obj.getClass()) {
        return false;
      }
      @Nullable final IndexMapKey other = (IndexMapKey) obj;
      return Arrays.equals(kernel, other.kernel) && Arrays.equals(output, other.output);
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + Arrays.hashCode(kernel);
      result = prime * result + Arrays.hashCode(output);
      return result;
    }
  }

  private static class LayerCacheLoader extends CacheLoader<IndexMapKey, RefMap<Coordinate, RefList<int[]>>> {
    @Override
    public RefMap<Coordinate, RefList<int[]>> load(@Nonnull final IndexMapKey key) {
      return getIndexMap(key);
    }
  }

  private static RefMap<Coordinate, RefList<int[]>> getIndexMap(@Nonnull IndexMapKey key) {
    Tensor tensor = new Tensor(key.output);
    try {
      final int[] ksize = key.kernel;
      return tensor.coordStream(false).map(x -> x.copy())
          .collect(RefCollectors.toMap(o -> o, o -> {
            @Nonnull Tensor blank = new Tensor(ksize);
            try {
              int[] coords = o.getCoords();
              return blank.coordStream(true).map(kernelCoord -> {
                int[] kernelCoords = kernelCoord.getCoords();
                @Nonnull final int[] r = new int[coords.length];
                for (int i = 0; i < coords.length; i++) {
                  r[i] = coords[i] * ksize[i] + kernelCoords[i];
                }
                return r;
              }).collect(RefCollectors.toList());
            } finally {
              blank.freeRef();
            }
          }));
    } finally {
      tensor.freeRef();
    }
  }
}
