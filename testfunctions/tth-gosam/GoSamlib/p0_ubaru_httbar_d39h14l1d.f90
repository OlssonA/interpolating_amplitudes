module     p0_ubaru_httbar_d39h14l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity14d39h14l1d.f90
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
      use p0_ubaru_httbar_abbrevd39h14
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(28) :: acd39
      complex(ki) :: brack
      acd39(1)=dotproduct(k1,qshift)
      acd39(2)=abb39(15)
      acd39(3)=dotproduct(k2,qshift)
      acd39(4)=dotproduct(qshift,qshift)
      acd39(5)=abb39(12)
      acd39(6)=dotproduct(qshift,spvak2k1)
      acd39(7)=dotproduct(qshift,spvak2l3)
      acd39(8)=abb39(13)
      acd39(9)=dotproduct(qshift,spvak2l4)
      acd39(10)=abb39(17)
      acd39(11)=dotproduct(qshift,spvak2l5)
      acd39(12)=abb39(26)
      acd39(13)=dotproduct(qshift,spval3l4)
      acd39(14)=abb39(19)
      acd39(15)=dotproduct(qshift,spval3l5)
      acd39(16)=abb39(16)
      acd39(17)=abb39(11)
      acd39(18)=abb39(10)
      acd39(19)=abb39(23)
      acd39(20)=dotproduct(qshift,spval3k1)
      acd39(21)=abb39(21)
      acd39(22)=abb39(18)
      acd39(23)=acd39(8)*acd39(7)
      acd39(24)=acd39(10)*acd39(9)
      acd39(25)=acd39(12)*acd39(11)
      acd39(26)=acd39(14)*acd39(13)
      acd39(27)=acd39(16)*acd39(15)
      acd39(23)=-acd39(17)+acd39(27)+acd39(26)+acd39(25)+acd39(24)+acd39(23)
      acd39(23)=acd39(6)*acd39(23)
      acd39(24)=-acd39(1)-acd39(3)
      acd39(24)=acd39(2)*acd39(24)
      acd39(25)=acd39(5)*acd39(4)
      acd39(26)=-acd39(18)*acd39(9)
      acd39(27)=-acd39(19)*acd39(11)
      acd39(28)=-acd39(21)*acd39(20)
      brack=acd39(22)+acd39(23)+acd39(24)+acd39(25)+acd39(26)+acd39(27)+acd39(2&
      &8)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd39h14
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(34) :: acd39
      complex(ki) :: brack
      acd39(1)=k1(iv1)
      acd39(2)=abb39(15)
      acd39(3)=k2(iv1)
      acd39(4)=qshift(iv1)
      acd39(5)=abb39(12)
      acd39(6)=spvak2k1(iv1)
      acd39(7)=dotproduct(qshift,spvak2l3)
      acd39(8)=abb39(13)
      acd39(9)=dotproduct(qshift,spvak2l4)
      acd39(10)=abb39(17)
      acd39(11)=dotproduct(qshift,spvak2l5)
      acd39(12)=abb39(26)
      acd39(13)=dotproduct(qshift,spval3l4)
      acd39(14)=abb39(19)
      acd39(15)=dotproduct(qshift,spval3l5)
      acd39(16)=abb39(16)
      acd39(17)=abb39(11)
      acd39(18)=spvak2l3(iv1)
      acd39(19)=dotproduct(qshift,spvak2k1)
      acd39(20)=spvak2l4(iv1)
      acd39(21)=abb39(10)
      acd39(22)=spvak2l5(iv1)
      acd39(23)=abb39(23)
      acd39(24)=spval3l4(iv1)
      acd39(25)=spval3l5(iv1)
      acd39(26)=spval3k1(iv1)
      acd39(27)=abb39(21)
      acd39(28)=acd39(20)*acd39(10)
      acd39(29)=acd39(22)*acd39(12)
      acd39(30)=acd39(18)*acd39(8)
      acd39(31)=acd39(24)*acd39(14)
      acd39(32)=acd39(25)*acd39(16)
      acd39(28)=acd39(32)+acd39(31)+acd39(30)+acd39(28)+acd39(29)
      acd39(28)=acd39(19)*acd39(28)
      acd39(29)=acd39(7)*acd39(8)
      acd39(30)=acd39(9)*acd39(10)
      acd39(31)=acd39(11)*acd39(12)
      acd39(32)=acd39(13)*acd39(14)
      acd39(33)=acd39(15)*acd39(16)
      acd39(29)=-acd39(17)+acd39(33)+acd39(32)+acd39(31)+acd39(30)+acd39(29)
      acd39(29)=acd39(6)*acd39(29)
      acd39(30)=-acd39(3)-acd39(1)
      acd39(30)=acd39(2)*acd39(30)
      acd39(31)=acd39(5)*acd39(4)
      acd39(32)=-acd39(21)*acd39(20)
      acd39(33)=-acd39(23)*acd39(22)
      acd39(34)=-acd39(27)*acd39(26)
      brack=acd39(28)+acd39(29)+acd39(30)+2.0_ki*acd39(31)+acd39(32)+acd39(33)+&
      &acd39(34)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd39h14
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(25) :: acd39
      complex(ki) :: brack
      acd39(1)=d(iv1,iv2)
      acd39(2)=abb39(12)
      acd39(3)=spvak2k1(iv1)
      acd39(4)=spvak2l3(iv2)
      acd39(5)=abb39(13)
      acd39(6)=spvak2l4(iv2)
      acd39(7)=abb39(17)
      acd39(8)=spvak2l5(iv2)
      acd39(9)=abb39(26)
      acd39(10)=spval3l4(iv2)
      acd39(11)=abb39(19)
      acd39(12)=spval3l5(iv2)
      acd39(13)=abb39(16)
      acd39(14)=spvak2k1(iv2)
      acd39(15)=spvak2l3(iv1)
      acd39(16)=spvak2l4(iv1)
      acd39(17)=spvak2l5(iv1)
      acd39(18)=spval3l4(iv1)
      acd39(19)=spval3l5(iv1)
      acd39(20)=acd39(4)*acd39(5)
      acd39(21)=acd39(6)*acd39(7)
      acd39(22)=acd39(8)*acd39(9)
      acd39(23)=acd39(10)*acd39(11)
      acd39(24)=acd39(12)*acd39(13)
      acd39(20)=acd39(24)+acd39(23)+acd39(22)+acd39(21)+acd39(20)
      acd39(20)=acd39(3)*acd39(20)
      acd39(21)=acd39(15)*acd39(5)
      acd39(22)=acd39(16)*acd39(7)
      acd39(23)=acd39(17)*acd39(9)
      acd39(24)=acd39(18)*acd39(11)
      acd39(25)=acd39(19)*acd39(13)
      acd39(21)=acd39(25)+acd39(24)+acd39(23)+acd39(22)+acd39(21)
      acd39(21)=acd39(14)*acd39(21)
      acd39(22)=acd39(2)*acd39(1)
      brack=acd39(20)+acd39(21)+2.0_ki*acd39(22)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd39h14
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
      qshift = -k3-k4-k5
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
end module     p0_ubaru_httbar_d39h14l1d
