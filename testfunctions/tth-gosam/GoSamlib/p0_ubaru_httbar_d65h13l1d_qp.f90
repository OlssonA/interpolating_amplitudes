module     p0_ubaru_httbar_d65h13l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity13d65h13l1d_qp.f90
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
      use p0_ubaru_httbar_abbrevd65h13_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(30) :: acd65
      complex(ki) :: brack
      acd65(1)=dotproduct(k2,qshift)
      acd65(2)=abb65(8)
      acd65(3)=dotproduct(l5,qshift)
      acd65(4)=abb65(22)
      acd65(5)=dotproduct(qshift,qshift)
      acd65(6)=abb65(10)
      acd65(7)=dotproduct(qshift,spvak1k2)
      acd65(8)=dotproduct(qshift,spvak2l3)
      acd65(9)=abb65(13)
      acd65(10)=dotproduct(qshift,spvak2l4)
      acd65(11)=abb65(9)
      acd65(12)=abb65(14)
      acd65(13)=abb65(25)
      acd65(14)=dotproduct(qshift,spvak1l3)
      acd65(15)=abb65(12)
      acd65(16)=dotproduct(qshift,spvak1l4)
      acd65(17)=abb65(11)
      acd65(18)=dotproduct(qshift,spvak1l5)
      acd65(19)=abb65(21)
      acd65(20)=dotproduct(qshift,spvak2l5)
      acd65(21)=abb65(23)
      acd65(22)=acd65(9)*acd65(7)
      acd65(22)=-acd65(12)+acd65(22)
      acd65(22)=acd65(8)*acd65(22)
      acd65(23)=acd65(11)*acd65(7)
      acd65(23)=-acd65(13)+acd65(23)
      acd65(23)=acd65(10)*acd65(23)
      acd65(24)=-acd65(2)*acd65(1)
      acd65(25)=-acd65(4)*acd65(3)
      acd65(26)=acd65(6)*acd65(5)
      acd65(27)=-acd65(15)*acd65(14)
      acd65(28)=-acd65(17)*acd65(16)
      acd65(29)=-acd65(19)*acd65(18)
      acd65(30)=-acd65(21)*acd65(20)
      brack=acd65(22)+acd65(23)+acd65(24)+acd65(25)+acd65(26)+acd65(27)+acd65(2&
      &8)+acd65(29)+acd65(30)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd65h13_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(34) :: acd65
      complex(ki) :: brack
      acd65(1)=k2(iv1)
      acd65(2)=abb65(8)
      acd65(3)=l5(iv1)
      acd65(4)=abb65(22)
      acd65(5)=qshift(iv1)
      acd65(6)=abb65(10)
      acd65(7)=spvak1k2(iv1)
      acd65(8)=dotproduct(qshift,spvak2l3)
      acd65(9)=abb65(13)
      acd65(10)=dotproduct(qshift,spvak2l4)
      acd65(11)=abb65(9)
      acd65(12)=spvak2l3(iv1)
      acd65(13)=dotproduct(qshift,spvak1k2)
      acd65(14)=abb65(14)
      acd65(15)=spvak2l4(iv1)
      acd65(16)=abb65(25)
      acd65(17)=spvak1l3(iv1)
      acd65(18)=abb65(12)
      acd65(19)=spvak1l4(iv1)
      acd65(20)=abb65(11)
      acd65(21)=spvak1l5(iv1)
      acd65(22)=abb65(21)
      acd65(23)=spvak2l5(iv1)
      acd65(24)=abb65(23)
      acd65(25)=acd65(8)*acd65(9)
      acd65(26)=acd65(10)*acd65(11)
      acd65(25)=acd65(26)+acd65(25)
      acd65(25)=acd65(7)*acd65(25)
      acd65(26)=acd65(13)*acd65(9)
      acd65(26)=-acd65(14)+acd65(26)
      acd65(26)=acd65(12)*acd65(26)
      acd65(27)=acd65(13)*acd65(11)
      acd65(27)=-acd65(16)+acd65(27)
      acd65(27)=acd65(15)*acd65(27)
      acd65(28)=-acd65(2)*acd65(1)
      acd65(29)=-acd65(4)*acd65(3)
      acd65(30)=acd65(6)*acd65(5)
      acd65(31)=-acd65(18)*acd65(17)
      acd65(32)=-acd65(20)*acd65(19)
      acd65(33)=-acd65(22)*acd65(21)
      acd65(34)=-acd65(24)*acd65(23)
      brack=acd65(25)+acd65(26)+acd65(27)+acd65(28)+acd65(29)+2.0_ki*acd65(30)+&
      &acd65(31)+acd65(32)+acd65(33)+acd65(34)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd65h13_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(13) :: acd65
      complex(ki) :: brack
      acd65(1)=d(iv1,iv2)
      acd65(2)=abb65(10)
      acd65(3)=spvak1k2(iv1)
      acd65(4)=spvak2l3(iv2)
      acd65(5)=abb65(13)
      acd65(6)=spvak2l4(iv2)
      acd65(7)=abb65(9)
      acd65(8)=spvak1k2(iv2)
      acd65(9)=spvak2l3(iv1)
      acd65(10)=spvak2l4(iv1)
      acd65(11)=acd65(4)*acd65(5)
      acd65(12)=acd65(6)*acd65(7)
      acd65(11)=acd65(12)+acd65(11)
      acd65(11)=acd65(3)*acd65(11)
      acd65(12)=acd65(9)*acd65(5)
      acd65(13)=acd65(10)*acd65(7)
      acd65(12)=acd65(13)+acd65(12)
      acd65(12)=acd65(8)*acd65(12)
      acd65(13)=acd65(2)*acd65(1)
      brack=acd65(11)+acd65(12)+2.0_ki*acd65(13)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd65h13_qp
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
      qshift = -k2
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
end module     p0_ubaru_httbar_d65h13l1d_qp
