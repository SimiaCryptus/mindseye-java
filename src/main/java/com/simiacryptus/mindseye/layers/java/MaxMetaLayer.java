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
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefComparator;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.Function;

@SuppressWarnings("serial")
public class MaxMetaLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxMetaLayer.class);

  public MaxMetaLayer() {
  }

  protected MaxMetaLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static MaxMetaLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new MaxMetaLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nullable final Result... inObj) {
    assert inObj != null;
    final Result input = inObj[0].addRef();
    ReferenceCounting.freeRefs(inObj);
    TensorList temp_40_0005 = input.getData();
    final int itemCnt = temp_40_0005.length();
    temp_40_0005.freeRef();
    TensorList temp_40_0006 = input.getData();
    final Tensor input0Tensor = temp_40_0006.get(0);
    temp_40_0006.freeRef();
    final int vectorSize = input0Tensor.length();
    @Nonnull final int[] indicies = new int[vectorSize];
    for (int i = 0; i < vectorSize; i++) {
      final int itemNumber = i;
      indicies[i] = RefUtil.get(RefIntStream.range(0, itemCnt).mapToObj(x -> x).max(
          RefComparator.comparing(RefUtil.wrapInterface((Function<? super Integer, ? extends Double>) dataIndex -> {
            TensorList temp_40_0007 = input.getData();
            Tensor tensor = temp_40_0007.get(dataIndex);
            temp_40_0007.freeRef();
            double temp_40_0003 = tensor.getData()[itemNumber];
            tensor.freeRef();
            return temp_40_0003;
          }, input.addRef()))));
    }
    try {
      Result.Accumulator accumulator = new Result.Accumulator() {
        {
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList data) {
          if (input.isAlive()) {
            @Nullable final Tensor delta = data.get(0);
            @Nonnull final Tensor feedback[] = new Tensor[itemCnt];
            RefArrays.parallelSetAll(RefUtil.addRefs(feedback),
                RefUtil.wrapInterface(i -> new Tensor(delta.getDimensions()), delta.addRef()));
            input0Tensor.coordStream(true)
                .forEach(RefUtil.wrapInterface((Consumer<? super Coordinate>) (inputCoord) -> {
                  feedback[indicies[inputCoord.getIndex()]].add(inputCoord, delta.get(inputCoord));
                }, delta.addRef(), RefUtil.addRefs(feedback)));
            delta.freeRef();
            @Nonnull
            TensorArray tensorArray = new TensorArray(RefUtil.addRefs(feedback));
            ReferenceCounting.freeRefs(feedback);
            input.accumulate(buffer == null ? null : buffer.addRef(), tensorArray);
          }
          data.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
        }
      };
      TensorArray data = new TensorArray(input0Tensor.mapIndex(RefUtil.wrapInterface((v, c) -> {
        TensorList temp_40_0008 = input.getData();
        Tensor tensor = temp_40_0008.get(indicies[c]);
        temp_40_0008.freeRef();
        double temp_40_0004 = tensor.getData()[c];
        tensor.freeRef();
        return temp_40_0004;
      }, input.addRef())));
      return new Result(data, accumulator) {
        {
          input.addRef();
        }
        @Override
        public boolean isAlive() {
          return input.isAlive();
        }

        @Override
        public void _free() {
          input.freeRef();
          super._free();
        }
      };
    } finally {
      input0Tensor.freeRef();
      input.freeRef();
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
  MaxMetaLayer addRef() {
    return (MaxMetaLayer) super.addRef();
  }
}
