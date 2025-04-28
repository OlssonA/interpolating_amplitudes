module     p0_ubaru_httbar_d1h14l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity14d1h14l1d.f90
   ! generator: buildfortran_d.py
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_util, only: cond, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, private :: iv0
   integer, private :: iv1
   integer, private :: iv2
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd1h14
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(20) :: acd1
      complex(ki) :: brack
      acd1(1)=dotproduct(qshift,spvak2k1)
      acd1(2)=dotproduct(qshift,spvak2l3)
      acd1(3)=abb1(9)
      acd1(4)=dotproduct(qshift,spvak2l4)
      acd1(5)=abb1(18)
      acd1(6)=dotproduct(qshift,spvak2l5)
      acd1(7)=abb1(14)
      acd1(8)=abb1(12)
      acd1(9)=abb1(21)
      acd1(10)=abb1(16)
      acd1(11)=dotproduct(qshift,spval3k1)
      acd1(12)=abb1(13)
      acd1(13)=abb1(17)
      acd1(14)=abb1(15)
      acd1(15)=abb1(10)
      acd1(16)=acd1(3)*acd1(2)
      acd1(17)=acd1(5)*acd1(4)
      acd1(18)=acd1(7)*acd1(6)
      acd1(16)=-acd1(8)+acd1(18)+acd1(16)+acd1(17)
      acd1(16)=acd1(1)*acd1(16)
      acd1(17)=-acd1(12)*acd1(6)
      acd1(17)=-acd1(14)+acd1(17)
      acd1(17)=acd1(11)*acd1(17)
      acd1(18)=-acd1(9)*acd1(2)
      acd1(19)=-acd1(10)*acd1(4)
      acd1(20)=-acd1(13)*acd1(6)
      brack=acd1(15)+acd1(16)+acd1(17)+acd1(18)+acd1(19)+acd1(20)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd1h14
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(25) :: acd1
      complex(ki) :: brack
      acd1(1)=spvak2k1(iv1)
      acd1(2)=dotproduct(qshift,spvak2l3)
      acd1(3)=abb1(9)
      acd1(4)=dotproduct(qshift,spvak2l4)
      acd1(5)=abb1(18)
      acd1(6)=dotproduct(qshift,spvak2l5)
      acd1(7)=abb1(14)
      acd1(8)=abb1(12)
      acd1(9)=spvak2l3(iv1)
      acd1(10)=dotproduct(qshift,spvak2k1)
      acd1(11)=abb1(21)
      acd1(12)=spvak2l4(iv1)
      acd1(13)=abb1(16)
      acd1(14)=spvak2l5(iv1)
      acd1(15)=dotproduct(qshift,spval3k1)
      acd1(16)=abb1(13)
      acd1(17)=abb1(17)
      acd1(18)=spval3k1(iv1)
      acd1(19)=abb1(15)
      acd1(20)=acd1(2)*acd1(3)
      acd1(21)=acd1(4)*acd1(5)
      acd1(20)=-acd1(8)+acd1(21)+acd1(20)
      acd1(20)=acd1(1)*acd1(20)
      acd1(21)=acd1(14)*acd1(10)
      acd1(22)=acd1(6)*acd1(1)
      acd1(21)=acd1(21)+acd1(22)
      acd1(21)=acd1(7)*acd1(21)
      acd1(22)=-acd1(15)*acd1(16)
      acd1(22)=-acd1(17)+acd1(22)
      acd1(22)=acd1(14)*acd1(22)
      acd1(23)=acd1(3)*acd1(10)
      acd1(23)=-acd1(11)+acd1(23)
      acd1(23)=acd1(9)*acd1(23)
      acd1(24)=acd1(5)*acd1(10)
      acd1(24)=-acd1(13)+acd1(24)
      acd1(24)=acd1(12)*acd1(24)
      acd1(25)=-acd1(16)*acd1(6)
      acd1(25)=-acd1(19)+acd1(25)
      acd1(25)=acd1(18)*acd1(25)
      brack=acd1(20)+acd1(21)+acd1(22)+acd1(23)+acd1(24)+acd1(25)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd1h14
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(18) :: acd1
      complex(ki) :: brack
      acd1(1)=spvak2k1(iv1)
      acd1(2)=spvak2l3(iv2)
      acd1(3)=abb1(9)
      acd1(4)=spvak2l4(iv2)
      acd1(5)=abb1(18)
      acd1(6)=spvak2l5(iv2)
      acd1(7)=abb1(14)
      acd1(8)=spvak2k1(iv2)
      acd1(9)=spvak2l3(iv1)
      acd1(10)=spvak2l4(iv1)
      acd1(11)=spvak2l5(iv1)
      acd1(12)=spval3k1(iv2)
      acd1(13)=abb1(13)
      acd1(14)=spval3k1(iv1)
      acd1(15)=acd1(7)*acd1(6)
      acd1(16)=acd1(2)*acd1(3)
      acd1(17)=acd1(4)*acd1(5)
      acd1(15)=acd1(17)+acd1(16)+acd1(15)
      acd1(15)=acd1(1)*acd1(15)
      acd1(16)=acd1(11)*acd1(7)
      acd1(17)=acd1(9)*acd1(3)
      acd1(18)=acd1(10)*acd1(5)
      acd1(16)=acd1(18)+acd1(17)+acd1(16)
      acd1(16)=acd1(8)*acd1(16)
      acd1(17)=-acd1(12)*acd1(11)
      acd1(18)=-acd1(14)*acd1(6)
      acd1(17)=acd1(18)+acd1(17)
      acd1(17)=acd1(13)*acd1(17)
      brack=acd1(15)+acd1(16)+acd1(17)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd1h14
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      integer, intent(in), optional :: i2
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = k5
      numerator = 0.0_ki
      deg = 0
      if(present(i1)) then
          iv1=i1
          deg=1
      else
          iv1=1
      end if
      if(present(i2)) then
          iv2=i2
          deg=2
      else
          iv2=1
      end if
      t1 = 0
      if(deg.eq.0) then
         numerator = cond(epspow.eq.t1,brack_1,Q,mu2)
         return
      end if
      if(deg.eq.1) then
         numerator = cond(epspow.eq.t1,brack_2,Q,mu2)
         return
      end if
      if(deg.eq.2) then
         numerator = cond(epspow.eq.t1,brack_3,Q,mu2)
         return
      end if
   end function derivative
!---#] function derivative:
end module     p0_ubaru_httbar_d1h14l1d
