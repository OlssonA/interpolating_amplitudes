module     p0_ubaru_httbar_d4h14l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity14d4h14l1d.f90
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
      use p0_ubaru_httbar_abbrevd4h14
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(28) :: acd4
      complex(ki) :: brack
      acd4(1)=dotproduct(k1,qshift)
      acd4(2)=abb4(24)
      acd4(3)=dotproduct(k2,qshift)
      acd4(4)=abb4(18)
      acd4(5)=dotproduct(l4,qshift)
      acd4(6)=dotproduct(qshift,qshift)
      acd4(7)=dotproduct(qshift,spvak2k1)
      acd4(8)=dotproduct(qshift,spvak2l3)
      acd4(9)=abb4(14)
      acd4(10)=dotproduct(qshift,spvak2l4)
      acd4(11)=abb4(10)
      acd4(12)=dotproduct(qshift,spvak2l5)
      acd4(13)=abb4(22)
      acd4(14)=dotproduct(qshift,spval3l4)
      acd4(15)=abb4(12)
      acd4(16)=abb4(11)
      acd4(17)=abb4(17)
      acd4(18)=abb4(15)
      acd4(19)=abb4(16)
      acd4(20)=abb4(25)
      acd4(21)=abb4(9)
      acd4(22)=acd4(9)*acd4(8)
      acd4(23)=acd4(11)*acd4(10)
      acd4(24)=acd4(13)*acd4(12)
      acd4(25)=acd4(15)*acd4(14)
      acd4(22)=-acd4(16)+acd4(25)+acd4(24)+acd4(23)+acd4(22)
      acd4(22)=acd4(7)*acd4(22)
      acd4(23)=acd4(6)-acd4(5)+acd4(3)
      acd4(23)=acd4(4)*acd4(23)
      acd4(24)=-acd4(2)*acd4(1)
      acd4(25)=-acd4(17)*acd4(8)
      acd4(26)=-acd4(18)*acd4(10)
      acd4(27)=-acd4(19)*acd4(12)
      acd4(28)=-acd4(20)*acd4(14)
      brack=acd4(21)+acd4(22)+acd4(23)+acd4(24)+acd4(25)+acd4(26)+acd4(27)+acd4&
      &(28)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd4h14
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(33) :: acd4
      complex(ki) :: brack
      acd4(1)=k1(iv1)
      acd4(2)=abb4(24)
      acd4(3)=k2(iv1)
      acd4(4)=abb4(18)
      acd4(5)=l4(iv1)
      acd4(6)=qshift(iv1)
      acd4(7)=spvak2k1(iv1)
      acd4(8)=dotproduct(qshift,spvak2l3)
      acd4(9)=abb4(14)
      acd4(10)=dotproduct(qshift,spvak2l4)
      acd4(11)=abb4(10)
      acd4(12)=dotproduct(qshift,spvak2l5)
      acd4(13)=abb4(22)
      acd4(14)=dotproduct(qshift,spval3l4)
      acd4(15)=abb4(12)
      acd4(16)=abb4(11)
      acd4(17)=spvak2l3(iv1)
      acd4(18)=dotproduct(qshift,spvak2k1)
      acd4(19)=abb4(17)
      acd4(20)=spvak2l4(iv1)
      acd4(21)=abb4(15)
      acd4(22)=spvak2l5(iv1)
      acd4(23)=abb4(16)
      acd4(24)=spval3l4(iv1)
      acd4(25)=abb4(25)
      acd4(26)=-acd4(17)*acd4(9)
      acd4(27)=-acd4(20)*acd4(11)
      acd4(28)=-acd4(22)*acd4(13)
      acd4(29)=-acd4(24)*acd4(15)
      acd4(26)=acd4(29)+acd4(28)+acd4(26)+acd4(27)
      acd4(26)=acd4(18)*acd4(26)
      acd4(27)=-acd4(8)*acd4(9)
      acd4(28)=-acd4(10)*acd4(11)
      acd4(29)=-acd4(12)*acd4(13)
      acd4(30)=-acd4(14)*acd4(15)
      acd4(27)=acd4(16)+acd4(30)+acd4(29)+acd4(28)+acd4(27)
      acd4(27)=acd4(7)*acd4(27)
      acd4(28)=-2.0_ki*acd4(6)+acd4(5)-acd4(3)
      acd4(28)=acd4(4)*acd4(28)
      acd4(29)=acd4(2)*acd4(1)
      acd4(30)=acd4(19)*acd4(17)
      acd4(31)=acd4(21)*acd4(20)
      acd4(32)=acd4(23)*acd4(22)
      acd4(33)=acd4(25)*acd4(24)
      brack=acd4(26)+acd4(27)+acd4(28)+acd4(29)+acd4(30)+acd4(31)+acd4(32)+acd4&
      &(33)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd4h14
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(21) :: acd4
      complex(ki) :: brack
      acd4(1)=d(iv1,iv2)
      acd4(2)=abb4(18)
      acd4(3)=spvak2k1(iv1)
      acd4(4)=spvak2l3(iv2)
      acd4(5)=abb4(14)
      acd4(6)=spvak2l4(iv2)
      acd4(7)=abb4(10)
      acd4(8)=spvak2l5(iv2)
      acd4(9)=abb4(22)
      acd4(10)=spval3l4(iv2)
      acd4(11)=abb4(12)
      acd4(12)=spvak2k1(iv2)
      acd4(13)=spvak2l3(iv1)
      acd4(14)=spvak2l4(iv1)
      acd4(15)=spvak2l5(iv1)
      acd4(16)=spval3l4(iv1)
      acd4(17)=acd4(4)*acd4(5)
      acd4(18)=acd4(6)*acd4(7)
      acd4(19)=acd4(8)*acd4(9)
      acd4(20)=acd4(10)*acd4(11)
      acd4(17)=acd4(20)+acd4(19)+acd4(18)+acd4(17)
      acd4(17)=acd4(3)*acd4(17)
      acd4(18)=acd4(13)*acd4(5)
      acd4(19)=acd4(14)*acd4(7)
      acd4(20)=acd4(15)*acd4(9)
      acd4(21)=acd4(16)*acd4(11)
      acd4(18)=acd4(21)+acd4(20)+acd4(19)+acd4(18)
      acd4(18)=acd4(12)*acd4(18)
      acd4(19)=acd4(2)*acd4(1)
      brack=acd4(17)+acd4(18)+2.0_ki*acd4(19)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd4h14
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
      qshift = k3+k4+k5
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
end module     p0_ubaru_httbar_d4h14l1d
