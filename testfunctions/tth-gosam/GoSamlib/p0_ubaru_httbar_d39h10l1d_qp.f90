module     p0_ubaru_httbar_d39h10l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity10d39h10l1d_qp.f90
   ! generator: buildfortran_d.py
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_util_qp, only: cond, d => metric_tensor
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
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd39h10_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(35) :: acd39
      complex(ki) :: brack
      acd39(1)=dotproduct(k1,qshift)
      acd39(2)=abb39(26)
      acd39(3)=dotproduct(k2,qshift)
      acd39(4)=dotproduct(qshift,spvak2k1)
      acd39(5)=abb39(12)
      acd39(6)=abb39(59)
      acd39(7)=dotproduct(qshift,qshift)
      acd39(8)=abb39(25)
      acd39(9)=dotproduct(qshift,spvak2l3)
      acd39(10)=abb39(16)
      acd39(11)=dotproduct(qshift,spval3k2)
      acd39(12)=abb39(15)
      acd39(13)=dotproduct(qshift,spval3l5)
      acd39(14)=abb39(14)
      acd39(15)=dotproduct(qshift,spval4l3)
      acd39(16)=abb39(11)
      acd39(17)=dotproduct(qshift,spval4l5)
      acd39(18)=abb39(13)
      acd39(19)=abb39(10)
      acd39(20)=abb39(35)
      acd39(21)=dotproduct(qshift,spvak2l5)
      acd39(22)=abb39(29)
      acd39(23)=dotproduct(qshift,spval3k1)
      acd39(24)=abb39(17)
      acd39(25)=dotproduct(qshift,spval4k1)
      acd39(26)=abb39(40)
      acd39(27)=abb39(19)
      acd39(28)=acd39(5)*acd39(3)
      acd39(29)=acd39(10)*acd39(9)
      acd39(30)=acd39(12)*acd39(11)
      acd39(31)=acd39(14)*acd39(13)
      acd39(32)=acd39(16)*acd39(15)
      acd39(33)=acd39(18)*acd39(17)
      acd39(28)=-acd39(19)+acd39(33)+acd39(32)+acd39(31)+acd39(30)+acd39(29)+ac&
      &d39(28)
      acd39(28)=acd39(4)*acd39(28)
      acd39(29)=-acd39(2)*acd39(1)
      acd39(30)=-acd39(6)*acd39(3)
      acd39(31)=acd39(8)*acd39(7)
      acd39(32)=-acd39(20)*acd39(9)
      acd39(33)=-acd39(22)*acd39(21)
      acd39(34)=-acd39(24)*acd39(23)
      acd39(35)=-acd39(26)*acd39(25)
      brack=acd39(27)+acd39(28)+acd39(29)+acd39(30)+acd39(31)+acd39(32)+acd39(3&
      &3)+acd39(34)+acd39(35)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd39h10_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(42) :: acd39
      complex(ki) :: brack
      acd39(1)=k1(iv1)
      acd39(2)=abb39(26)
      acd39(3)=k2(iv1)
      acd39(4)=dotproduct(qshift,spvak2k1)
      acd39(5)=abb39(12)
      acd39(6)=abb39(59)
      acd39(7)=qshift(iv1)
      acd39(8)=abb39(25)
      acd39(9)=spvak2k1(iv1)
      acd39(10)=dotproduct(k2,qshift)
      acd39(11)=dotproduct(qshift,spvak2l3)
      acd39(12)=abb39(16)
      acd39(13)=dotproduct(qshift,spval3k2)
      acd39(14)=abb39(15)
      acd39(15)=dotproduct(qshift,spval3l5)
      acd39(16)=abb39(14)
      acd39(17)=dotproduct(qshift,spval4l3)
      acd39(18)=abb39(11)
      acd39(19)=dotproduct(qshift,spval4l5)
      acd39(20)=abb39(13)
      acd39(21)=abb39(10)
      acd39(22)=spvak2l3(iv1)
      acd39(23)=abb39(35)
      acd39(24)=spval3k2(iv1)
      acd39(25)=spval3l5(iv1)
      acd39(26)=spval4l3(iv1)
      acd39(27)=spval4l5(iv1)
      acd39(28)=spvak2l5(iv1)
      acd39(29)=abb39(29)
      acd39(30)=spval3k1(iv1)
      acd39(31)=abb39(17)
      acd39(32)=spval4k1(iv1)
      acd39(33)=abb39(40)
      acd39(34)=acd39(5)*acd39(3)
      acd39(35)=acd39(22)*acd39(12)
      acd39(36)=acd39(24)*acd39(14)
      acd39(37)=acd39(25)*acd39(16)
      acd39(38)=acd39(26)*acd39(18)
      acd39(39)=acd39(27)*acd39(20)
      acd39(34)=acd39(39)+acd39(38)+acd39(37)+acd39(36)+acd39(34)+acd39(35)
      acd39(34)=acd39(4)*acd39(34)
      acd39(35)=acd39(10)*acd39(5)
      acd39(36)=acd39(11)*acd39(12)
      acd39(37)=acd39(13)*acd39(14)
      acd39(38)=acd39(15)*acd39(16)
      acd39(39)=acd39(17)*acd39(18)
      acd39(40)=acd39(19)*acd39(20)
      acd39(35)=-acd39(21)+acd39(40)+acd39(39)+acd39(38)+acd39(37)+acd39(36)+ac&
      &d39(35)
      acd39(35)=acd39(9)*acd39(35)
      acd39(36)=-acd39(2)*acd39(1)
      acd39(37)=-acd39(6)*acd39(3)
      acd39(38)=acd39(8)*acd39(7)
      acd39(39)=-acd39(23)*acd39(22)
      acd39(40)=-acd39(29)*acd39(28)
      acd39(41)=-acd39(31)*acd39(30)
      acd39(42)=-acd39(33)*acd39(32)
      brack=acd39(34)+acd39(35)+acd39(36)+acd39(37)+2.0_ki*acd39(38)+acd39(39)+&
      &acd39(40)+acd39(41)+acd39(42)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd39h10_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(29) :: acd39
      complex(ki) :: brack
      acd39(1)=d(iv1,iv2)
      acd39(2)=abb39(25)
      acd39(3)=k2(iv1)
      acd39(4)=spvak2k1(iv2)
      acd39(5)=abb39(12)
      acd39(6)=k2(iv2)
      acd39(7)=spvak2k1(iv1)
      acd39(8)=spvak2l3(iv2)
      acd39(9)=abb39(16)
      acd39(10)=spval3k2(iv2)
      acd39(11)=abb39(15)
      acd39(12)=spval3l5(iv2)
      acd39(13)=abb39(14)
      acd39(14)=spval4l3(iv2)
      acd39(15)=abb39(11)
      acd39(16)=spval4l5(iv2)
      acd39(17)=abb39(13)
      acd39(18)=spvak2l3(iv1)
      acd39(19)=spval3k2(iv1)
      acd39(20)=spval3l5(iv1)
      acd39(21)=spval4l3(iv1)
      acd39(22)=spval4l5(iv1)
      acd39(23)=acd39(3)*acd39(5)
      acd39(24)=acd39(18)*acd39(9)
      acd39(25)=acd39(19)*acd39(11)
      acd39(26)=acd39(20)*acd39(13)
      acd39(27)=acd39(21)*acd39(15)
      acd39(28)=acd39(22)*acd39(17)
      acd39(23)=acd39(28)+acd39(27)+acd39(26)+acd39(25)+acd39(24)+acd39(23)
      acd39(23)=acd39(4)*acd39(23)
      acd39(24)=acd39(6)*acd39(5)
      acd39(25)=acd39(8)*acd39(9)
      acd39(26)=acd39(10)*acd39(11)
      acd39(27)=acd39(12)*acd39(13)
      acd39(28)=acd39(14)*acd39(15)
      acd39(29)=acd39(16)*acd39(17)
      acd39(24)=acd39(29)+acd39(28)+acd39(27)+acd39(26)+acd39(25)+acd39(24)
      acd39(24)=acd39(7)*acd39(24)
      acd39(25)=acd39(2)*acd39(1)
      brack=acd39(23)+acd39(24)+2.0_ki*acd39(25)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd39h10_qp
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
end module     p0_ubaru_httbar_d39h10l1d_qp
