module     p0_ubaru_httbar_d84h10l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity10d84h10l1d_qp.f90
   ! generator: buildfortran_d.py
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_util_qp, only: cond, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, private :: iv0
   integer, private :: iv1
   integer, private :: iv2
   integer, private :: iv3
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd84h10_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(35) :: acd84
      complex(ki) :: brack
      acd84(1)=dotproduct(k2,qshift)
      acd84(2)=abb84(13)
      acd84(3)=dotproduct(qshift,qshift)
      acd84(4)=abb84(14)
      acd84(5)=dotproduct(qshift,spvak2k1)
      acd84(6)=abb84(6)
      acd84(7)=abb84(8)
      acd84(8)=dotproduct(l4,qshift)
      acd84(9)=abb84(15)
      acd84(10)=abb84(22)
      acd84(11)=abb84(10)
      acd84(12)=dotproduct(qshift,spvak1k2)
      acd84(13)=abb84(27)
      acd84(14)=abb84(9)
      acd84(15)=abb84(7)
      acd84(16)=dotproduct(qshift,spvak2l4)
      acd84(17)=dotproduct(qshift,spval4k1)
      acd84(18)=abb84(17)
      acd84(19)=abb84(26)
      acd84(20)=dotproduct(qshift,spvak2l5)
      acd84(21)=dotproduct(qshift,spval5k2)
      acd84(22)=abb84(24)
      acd84(23)=abb84(23)
      acd84(24)=dotproduct(qshift,spval4l5)
      acd84(25)=abb84(25)
      acd84(26)=-acd84(3)*acd84(4)
      acd84(27)=acd84(5)*acd84(6)
      acd84(28)=acd84(1)*acd84(2)
      acd84(26)=acd84(28)+acd84(27)-acd84(7)+acd84(26)
      acd84(26)=acd84(1)*acd84(26)
      acd84(27)=acd84(12)*acd84(13)
      acd84(28)=-acd84(3)*acd84(10)
      acd84(27)=acd84(28)-acd84(14)+acd84(27)
      acd84(27)=acd84(5)*acd84(27)
      acd84(28)=-acd84(13)*acd84(21)
      acd84(28)=acd84(28)-acd84(22)
      acd84(28)=acd84(20)*acd84(28)
      acd84(29)=-acd84(10)*acd84(17)
      acd84(29)=acd84(29)-acd84(18)
      acd84(29)=acd84(16)*acd84(29)
      acd84(30)=-acd84(24)*acd84(25)
      acd84(31)=-acd84(8)*acd84(9)
      acd84(32)=-acd84(21)*acd84(23)
      acd84(33)=-acd84(17)*acd84(19)
      acd84(34)=-acd84(12)*acd84(15)
      acd84(35)=acd84(3)*acd84(11)
      brack=acd84(26)+acd84(27)+acd84(28)+acd84(29)+acd84(30)+acd84(31)+acd84(3&
      &2)+acd84(33)+acd84(34)+acd84(35)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd84h10_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(45) :: acd84
      complex(ki) :: brack
      acd84(1)=k2(iv1)
      acd84(2)=dotproduct(k2,qshift)
      acd84(3)=abb84(13)
      acd84(4)=dotproduct(qshift,qshift)
      acd84(5)=abb84(14)
      acd84(6)=dotproduct(qshift,spvak2k1)
      acd84(7)=abb84(6)
      acd84(8)=abb84(8)
      acd84(9)=l4(iv1)
      acd84(10)=abb84(15)
      acd84(11)=qshift(iv1)
      acd84(12)=abb84(22)
      acd84(13)=abb84(10)
      acd84(14)=spvak2k1(iv1)
      acd84(15)=dotproduct(qshift,spvak1k2)
      acd84(16)=abb84(27)
      acd84(17)=abb84(9)
      acd84(18)=spvak1k2(iv1)
      acd84(19)=abb84(7)
      acd84(20)=spvak2l4(iv1)
      acd84(21)=dotproduct(qshift,spval4k1)
      acd84(22)=abb84(17)
      acd84(23)=spval4k1(iv1)
      acd84(24)=dotproduct(qshift,spvak2l4)
      acd84(25)=abb84(26)
      acd84(26)=spvak2l5(iv1)
      acd84(27)=dotproduct(qshift,spval5k2)
      acd84(28)=abb84(24)
      acd84(29)=spval5k2(iv1)
      acd84(30)=dotproduct(qshift,spvak2l5)
      acd84(31)=abb84(23)
      acd84(32)=spval4l5(iv1)
      acd84(33)=abb84(25)
      acd84(34)=acd84(23)*acd84(24)
      acd84(35)=acd84(20)*acd84(21)
      acd84(36)=2.0_ki*acd84(11)
      acd84(37)=acd84(6)*acd84(36)
      acd84(38)=acd84(14)*acd84(4)
      acd84(34)=acd84(38)+acd84(37)+acd84(34)+acd84(35)
      acd84(34)=acd84(12)*acd84(34)
      acd84(35)=acd84(29)*acd84(30)
      acd84(37)=acd84(26)*acd84(27)
      acd84(38)=-acd84(6)*acd84(18)
      acd84(35)=acd84(38)+acd84(35)+acd84(37)
      acd84(35)=acd84(16)*acd84(35)
      acd84(37)=acd84(4)*acd84(5)
      acd84(38)=-acd84(6)*acd84(7)
      acd84(39)=2.0_ki*acd84(2)
      acd84(39)=-acd84(3)*acd84(39)
      acd84(37)=acd84(39)+acd84(38)+acd84(8)+acd84(37)
      acd84(37)=acd84(1)*acd84(37)
      acd84(38)=acd84(2)*acd84(5)
      acd84(38)=acd84(38)-acd84(13)
      acd84(36)=acd84(36)*acd84(38)
      acd84(38)=-acd84(2)*acd84(7)
      acd84(39)=-acd84(16)*acd84(15)
      acd84(38)=acd84(39)+acd84(17)+acd84(38)
      acd84(38)=acd84(14)*acd84(38)
      acd84(39)=acd84(32)*acd84(33)
      acd84(40)=acd84(9)*acd84(10)
      acd84(41)=acd84(29)*acd84(31)
      acd84(42)=acd84(26)*acd84(28)
      acd84(43)=acd84(23)*acd84(25)
      acd84(44)=acd84(20)*acd84(22)
      acd84(45)=acd84(18)*acd84(19)
      brack=acd84(34)+acd84(35)+acd84(36)+acd84(37)+acd84(38)+acd84(39)+acd84(4&
      &0)+acd84(41)+acd84(42)+acd84(43)+acd84(44)+acd84(45)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd84h10_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(31) :: acd84
      complex(ki) :: brack
      acd84(1)=d(iv1,iv2)
      acd84(2)=dotproduct(k2,qshift)
      acd84(3)=abb84(14)
      acd84(4)=dotproduct(qshift,spvak2k1)
      acd84(5)=abb84(22)
      acd84(6)=abb84(10)
      acd84(7)=k2(iv1)
      acd84(8)=k2(iv2)
      acd84(9)=abb84(13)
      acd84(10)=qshift(iv2)
      acd84(11)=spvak2k1(iv2)
      acd84(12)=abb84(6)
      acd84(13)=qshift(iv1)
      acd84(14)=spvak2k1(iv1)
      acd84(15)=spvak1k2(iv2)
      acd84(16)=abb84(27)
      acd84(17)=spvak1k2(iv1)
      acd84(18)=spvak2l4(iv1)
      acd84(19)=spval4k1(iv2)
      acd84(20)=spvak2l4(iv2)
      acd84(21)=spval4k1(iv1)
      acd84(22)=spvak2l5(iv1)
      acd84(23)=spval5k2(iv2)
      acd84(24)=spvak2l5(iv2)
      acd84(25)=spval5k2(iv1)
      acd84(26)=acd84(15)*acd84(14)
      acd84(27)=acd84(17)*acd84(11)
      acd84(28)=-acd84(23)*acd84(22)
      acd84(29)=-acd84(25)*acd84(24)
      acd84(26)=acd84(29)+acd84(28)+acd84(27)+acd84(26)
      acd84(26)=acd84(16)*acd84(26)
      acd84(27)=-acd84(7)*acd84(3)
      acd84(28)=-acd84(14)*acd84(5)
      acd84(27)=acd84(27)+acd84(28)
      acd84(27)=acd84(10)*acd84(27)
      acd84(28)=-acd84(8)*acd84(3)
      acd84(29)=-acd84(11)*acd84(5)
      acd84(28)=acd84(28)+acd84(29)
      acd84(28)=acd84(13)*acd84(28)
      acd84(29)=acd84(9)*acd84(8)*acd84(7)
      acd84(27)=acd84(29)+acd84(27)+acd84(28)
      acd84(28)=-acd84(19)*acd84(18)
      acd84(29)=-acd84(21)*acd84(20)
      acd84(28)=acd84(29)+acd84(28)
      acd84(28)=acd84(5)*acd84(28)
      acd84(29)=-acd84(2)*acd84(3)
      acd84(30)=-acd84(4)*acd84(5)
      acd84(29)=acd84(6)+acd84(30)+acd84(29)
      acd84(30)=2.0_ki*acd84(1)
      acd84(29)=acd84(30)*acd84(29)
      acd84(30)=acd84(11)*acd84(7)
      acd84(31)=acd84(14)*acd84(8)
      acd84(30)=acd84(30)+acd84(31)
      acd84(30)=acd84(12)*acd84(30)
      brack=acd84(26)+2.0_ki*acd84(27)+acd84(28)+acd84(29)+acd84(30)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd84h10_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(15) :: acd84
      complex(ki) :: brack
      acd84(1)=d(iv1,iv2)
      acd84(2)=k2(iv3)
      acd84(3)=abb84(14)
      acd84(4)=spvak2k1(iv3)
      acd84(5)=abb84(22)
      acd84(6)=d(iv1,iv3)
      acd84(7)=k2(iv2)
      acd84(8)=spvak2k1(iv2)
      acd84(9)=d(iv2,iv3)
      acd84(10)=k2(iv1)
      acd84(11)=spvak2k1(iv1)
      acd84(12)=acd84(2)*acd84(1)
      acd84(13)=acd84(7)*acd84(6)
      acd84(14)=acd84(10)*acd84(9)
      acd84(12)=acd84(14)+acd84(13)+acd84(12)
      acd84(12)=acd84(3)*acd84(12)
      acd84(13)=acd84(4)*acd84(1)
      acd84(14)=acd84(8)*acd84(6)
      acd84(15)=acd84(11)*acd84(9)
      acd84(13)=acd84(15)+acd84(14)+acd84(13)
      acd84(13)=acd84(5)*acd84(13)
      acd84(12)=acd84(13)+acd84(12)
      brack=2.0_ki*acd84(12)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd84h10_qp
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      integer, intent(in), optional :: i2
      integer, intent(in), optional :: i3
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = k2
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
      if(present(i3)) then
          iv3=i3
          deg=3
      else
          iv3=1
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
      if(deg.eq.3) then
         numerator = cond(epspow.eq.t1,brack_4,Q,mu2)
         return
      end if
   end function derivative
!---#] function derivative:
end module     p0_ubaru_httbar_d84h10l1d_qp
