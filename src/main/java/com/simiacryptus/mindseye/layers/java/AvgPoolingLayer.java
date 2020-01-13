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

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.JsonUtil;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.Map.Entry;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.function.IntFunction;
import java.util.function.ToDoubleFunction;

@SuppressWarnings("serial")
public class AvgPoolingLayer extends LayerBase {

  public static final LoadingCache<AvgPoolingLayer.IndexMapKey, RefMap<Coordinate, RefList<int[]>>> indexMapCache = CacheBuilder
      .newBuilder().build(new LayerCacheLoader());
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

  @SuppressWarnings("unused")
  public static AvgPoolingLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new AvgPoolingLayer(json, JsonUtil.getIntArray(json.getAsJsonArray("heapCopy")));
  }

  public static @SuppressWarnings("unused") AvgPoolingLayer[] addRefs(AvgPoolingLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(AvgPoolingLayer::addRef)
        .toArray((x) -> new AvgPoolingLayer[x]);
  }

  public static @SuppressWarnings("unused") AvgPoolingLayer[][] addRefs(AvgPoolingLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(AvgPoolingLayer::addRefs)
        .toArray((x) -> new AvgPoolingLayer[x][]);
  }

  private static synchronized RefMap<Coordinate, RefList<int[]>> getCoordMap(final int[] kernelDims,
      final int[] outDims) {
    try {
      return AvgPoolingLayer.indexMapCache.get(new AvgPoolingLayer.IndexMapKey(kernelDims, outDims));
    } catch (@Nonnull final ExecutionException e) {
      throw new RuntimeException(e);
    }
  }

  @Nonnull
  @SuppressWarnings("unchecked")
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final int kernelSize = Tensor.length(kernelDims);
    final TensorList data = inObj[0].getData();
    @Nonnull
    final int[] inputDims = data.getDimensions();
    final int[] newDims = RefIntStream.range(0, inputDims.length).map(i -> {
      assert 0 == inputDims[i] % kernelDims[i] : inputDims[i] + ":" + kernelDims[i];
      return inputDims[i] / kernelDims[i];
    }).toArray();
    final RefMap<Coordinate, RefList<int[]>> coordMap = AvgPoolingLayer.getCoordMap(kernelDims, newDims);
    final Tensor[] outputValues = RefIntStream.range(0, data.length())
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          @Nullable
          final Tensor input = data.get(dataIndex);
          @Nonnull
          final Tensor output = new Tensor(newDims);
          for (@Nonnull
          final Entry<Coordinate, RefList<int[]>> entry : coordMap.entrySet()) {
            RefList<int[]> temp_30_0006 = entry.getValue();
            double sum = temp_30_0006.stream()
                .mapToDouble(
                    RefUtil.wrapInterface((ToDoubleFunction<? super int[]>) inputCoord -> input.get(inputCoord),
                        input == null ? null : input.addRef()))
                .sum();
            if (null != temp_30_0006)
              temp_30_0006.freeRef();
            if (Double.isFinite(sum)) {
              output.add(entry.getKey(), sum / kernelSize);
            }
          }
          if (null != input)
            input.freeRef();
          return output;
        }, data == null ? null : data.addRef(), coordMap == null ? null : coordMap.addRef()))
        .toArray(i -> new Tensor[i]);
    if (null != data)
      data.freeRef();
    try {
      try {
        try {
          return new Result(new TensorArray(Tensor.addRefs(outputValues)), new Result.Accumulator() {
            {
              Result.addRefs(inObj);
              coordMap.addRef();
            }

            @Override
            public void accept(DeltaSet<UUID> buffer, TensorList delta) {
              if (inObj[0].isAlive()) {
                final Tensor[] passback = RefIntStream.range(0, delta.length())
                    .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                      @Nullable
                      Tensor tensor = delta.get(dataIndex);
                      @Nonnull
                      final Tensor backSignal = new Tensor(inputDims);
                      for (@Nonnull
                      final Entry<Coordinate, RefList<int[]>> outputMapping : coordMap.entrySet()) {
                        final double outputValue = tensor.get(outputMapping.getKey());
                        for (@Nonnull
                        final int[] inputCoord : outputMapping.getValue()) {
                          backSignal.add(inputCoord, outputValue / kernelSize);
                        }
                      }
                      if (null != tensor)
                        tensor.freeRef();
                      return backSignal;
                    }, delta == null ? null : delta.addRef(), coordMap == null ? null : coordMap.addRef()))
                    .toArray(i -> new Tensor[i]);
                @Nonnull
                TensorArray tensorArray = new TensorArray(Tensor.addRefs(passback));
                if (null != passback)
                  ReferenceCounting.freeRefs(passback);
                inObj[0].accumulate(buffer == null ? null : buffer.addRef(), tensorArray == null ? null : tensorArray);
              }
              if (null != delta)
                delta.freeRef();
              if (null != buffer)
                buffer.freeRef();
            }

            public @SuppressWarnings("unused") void _free() {
              ReferenceCounting.freeRefs(inObj);
              RefUtil.freeRef(coordMap);
            }
          }) {

            {
              Result.addRefs(inObj);
            }

            @Override
            public boolean isAlive() {
              return inObj[0].isAlive();
            }

            public void _free() {
              ReferenceCounting.freeRefs(inObj);
            }
          };
        } finally {
          ReferenceCounting.freeRefs(inObj);
        }
      } finally {
        if (null != outputValues)
          ReferenceCounting.freeRefs(outputValues);
      }
    } finally {
      if (null != coordMap)
        coordMap.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJsonStub();
    json.add("heapCopy", JsonUtil.getJson(kernelDims));
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") AvgPoolingLayer addRef() {
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

    public IndexMapKey(@Nonnull final Tensor kernel, final Tensor input, @Nonnull final Tensor output) {
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
      @Nullable
      final AvgPoolingLayer.IndexMapKey other = (AvgPoolingLayer.IndexMapKey) obj;
      if (!RefArrays.equals(kernel, other.kernel)) {
        return false;
      }
      return RefArrays.equals(output, other.output);
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + RefArrays.hashCode(kernel);
      result = prime * result + RefArrays.hashCode(output);
      return result;
    }
  }

  private static class LayerCacheLoader extends CacheLoader<IndexMapKey, RefMap<Coordinate, RefList<int[]>>> {
    @Override
    public RefMap<Coordinate, RefList<int[]>> load(@NotNull final IndexMapKey key) {
      final int[] ksize = key.kernel;
      Tensor tensor = new Tensor(key.output);
      RefMap<Coordinate, RefList<int[]>> temp_30_0003 = tensor.coordStream(true)
          .collect(RefCollectors.toMap(o -> o, o -> {
            @Nonnull
            Tensor blank = new Tensor(ksize);
            RefList<int[]> temp_30_0004 = blank.coordStream(true).map(kernelCoord -> {
              int[] coords = o.getCoords();
              @Nonnull
              final int[] r = new int[coords.length];
              for (int i = 0; i < coords.length; i++) {
                r[i] = coords[i] * ksize[i] + kernelCoord.getCoords()[i];
              }
              return r;
            }).collect(RefCollectors.toList());
            if (null != blank)
              blank.freeRef();
            return temp_30_0004;
          }));
      if (null != tensor)
        tensor.freeRef();
      return temp_30_0003;
    }
  }
}
