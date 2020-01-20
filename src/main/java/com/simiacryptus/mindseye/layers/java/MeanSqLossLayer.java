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

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class MeanSqLossLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MeanSqLossLayer.class);

  public MeanSqLossLayer() {
  }

  protected MeanSqLossLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static MeanSqLossLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new MeanSqLossLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (2 != inObj.length) {
      ReferenceCounting.freeRefs(inObj);
      throw new IllegalArgumentException();
    }
    TensorList temp_20_0011 = inObj[0].getData();
    final int leftLength = temp_20_0011.length();
    temp_20_0011.freeRef();
    TensorList temp_20_0012 = inObj[1].getData();
    final int rightLength = temp_20_0012.length();
    temp_20_0012.freeRef();
    if (leftLength != rightLength && leftLength != 1 && rightLength != 1) {
      ReferenceCounting.freeRefs(inObj);
      throw new IllegalArgumentException(leftLength + " != " + rightLength);
    }
    @Nonnull final Tensor diffs[] = new Tensor[leftLength];
    try {
      return new Result(new TensorArray(RefIntStream.range(0, leftLength)
          .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
            TensorList temp_20_0013 = inObj[0].getData();
            @Nullable final Tensor a = temp_20_0013.get(1 == leftLength ? 0 : dataIndex);
            temp_20_0013.freeRef();
            TensorList temp_20_0014 = inObj[1].getData();
            @Nullable final Tensor b = temp_20_0014.get(1 == rightLength ? 0 : dataIndex);
            temp_20_0014.freeRef();
            if (a.length() != b.length()) {
              IllegalArgumentException temp_20_0003 = new IllegalArgumentException(RefString.format("%s != %s",
                  RefArrays.toString(a.getDimensions()), RefArrays.toString(b.getDimensions())));
              a.freeRef();
              b.freeRef();
              throw temp_20_0003;
            }
            @Nonnull final Tensor r = a.minus(b.addRef());
            b.freeRef();
            a.freeRef();
            Tensor temp_20_0004 = new Tensor(new double[]{r.sumSq() / r.length()}, 1);
            RefUtil.set((diffs), dataIndex, r);
            return temp_20_0004;
          }, RefUtil.addRefs(diffs), RefUtil.addRefs(inObj))).toArray(i -> new Tensor[i])), new Result.Accumulator() {
        {
          RefUtil.addRefs(inObj);
          RefUtil.addRefs(diffs);
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList data) {
          if (inObj[0].isAlive()) {
            RefList<Tensor> temp_20_0015 = RefIntStream.range(0, data.length()).parallel()
                .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                  @Nullable
                  Tensor tensor = data.get(dataIndex);
                  Tensor diff = diffs[dataIndex].addRef();
                  Tensor temp_20_0005 = diff.scale(tensor.get(0) * 2.0 / diff.length());
                  diff.freeRef();
                  tensor.freeRef();
                  return temp_20_0005;
                }, data.addRef(), RefUtil.addRefs(diffs))).collect(RefCollectors.toList());
            RefStream<Tensor> tensorStream = temp_20_0015.stream();
            temp_20_0015.freeRef();
            if (1 == leftLength) {
              tensorStream = RefStream.of(RefUtil.get(tensorStream.reduce((a, b) -> {
                Tensor temp_20_0006 = a.addAndFree(b == null ? null : b.addRef());
                if (null != b)
                  b.freeRef();
                a.freeRef();
                return temp_20_0006;
              })));
            }
            @Nonnull final TensorList array = new TensorArray(tensorStream.toArray(i -> new Tensor[i]));
            inObj[0].accumulate(buffer == null ? null : buffer.addRef(), array);
          }
          if (inObj[1].isAlive()) {
            RefList<Tensor> temp_20_0016 = RefIntStream.range(0, data.length()).parallel()
                .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                  @Nullable
                  Tensor tensor = data.get(dataIndex);
                  Tensor temp_20_0007 = diffs[dataIndex].scale(tensor.get(0) * 2.0 / diffs[dataIndex].length());
                  tensor.freeRef();
                  return temp_20_0007;
                }, data.addRef(), RefUtil.addRefs(diffs))).collect(RefCollectors.toList());
            RefStream<Tensor> tensorStream = temp_20_0016.stream();
            temp_20_0016.freeRef();
            if (1 == rightLength) {
              tensorStream = RefStream.of(RefUtil.get(tensorStream.reduce((a, b) -> {
                Tensor temp_20_0008 = a.addAndFree(b == null ? null : b.addRef());
                if (null != b)
                  b.freeRef();
                a.freeRef();
                return temp_20_0008;
              })));
            }
            @Nonnull final TensorList array = new TensorArray(tensorStream.map(x -> {
              Tensor temp_20_0009 = x.scale(-1);
              x.freeRef();
              return temp_20_0009;
            }).toArray(i -> new Tensor[i]));
            inObj[1].accumulate(buffer == null ? null : buffer.addRef(), array);
          }
          data.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          ReferenceCounting.freeRefs(inObj);
          RefUtil.freeRefs(diffs);
        }
      }) {

        {
          RefUtil.addRefs(inObj);
        }

        @Override
        public boolean isAlive() {
          return inObj[0].isAlive() || inObj[1].isAlive();
        }

        public void _free() {
          ReferenceCounting.freeRefs(inObj);
          super._free();
        }
      };
    } finally {
      ReferenceCounting.freeRefs(inObj);
      ReferenceCounting.freeRefs(diffs);
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  MeanSqLossLayer addRef() {
    return (MeanSqLossLayer) super.addRef();
  }
}
