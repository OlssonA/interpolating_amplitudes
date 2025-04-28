module     p2_gg_httbar_d44h12l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d44h12l1d_qp.f90
   ! generator: buildfortran_d.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond, d => metric_tensor
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
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd44h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd44
      complex(ki) :: brack
      acd44(1)=abb44(14)
      brack=acd44(1)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd44h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(87) :: acd44
      complex(ki) :: brack
      acd44(1)=k1(iv1)
      acd44(2)=abb44(16)
      acd44(3)=k2(iv1)
      acd44(4)=abb44(18)
      acd44(5)=l4(iv1)
      acd44(6)=abb44(20)
      acd44(7)=spvak1k2(iv1)
      acd44(8)=abb44(23)
      acd44(9)=spvak1l4(iv1)
      acd44(10)=abb44(34)
      acd44(11)=spvak1l5(iv1)
      acd44(12)=abb44(22)
      acd44(13)=spvak2k1(iv1)
      acd44(14)=abb44(15)
      acd44(15)=spvak2l4(iv1)
      acd44(16)=abb44(42)
      acd44(17)=spval4k1(iv1)
      acd44(18)=abb44(24)
      acd44(19)=spval4k2(iv1)
      acd44(20)=abb44(41)
      acd44(21)=spval4l5(iv1)
      acd44(22)=abb44(35)
      acd44(23)=spval5k1(iv1)
      acd44(24)=abb44(53)
      acd44(25)=spval5k2(iv1)
      acd44(26)=abb44(25)
      acd44(27)=spval5l4(iv1)
      acd44(28)=abb44(54)
      acd44(29)=spvak1e1(iv1)
      acd44(30)=abb44(21)
      acd44(31)=spvae1k1(iv1)
      acd44(32)=abb44(49)
      acd44(33)=spvak1e2(iv1)
      acd44(34)=abb44(47)
      acd44(35)=spvae2k1(iv1)
      acd44(36)=abb44(50)
      acd44(37)=spvak2e1(iv1)
      acd44(38)=abb44(61)
      acd44(39)=spvae1k2(iv1)
      acd44(40)=abb44(58)
      acd44(41)=spvae2k2(iv1)
      acd44(42)=abb44(52)
      acd44(43)=spval4e1(iv1)
      acd44(44)=abb44(38)
      acd44(45)=spvae1l4(iv1)
      acd44(46)=abb44(32)
      acd44(47)=spval4e2(iv1)
      acd44(48)=abb44(31)
      acd44(49)=spvae2l4(iv1)
      acd44(50)=abb44(45)
      acd44(51)=spval5e1(iv1)
      acd44(52)=abb44(27)
      acd44(53)=spvae1l5(iv1)
      acd44(54)=abb44(40)
      acd44(55)=spvae1e2(iv1)
      acd44(56)=abb44(33)
      acd44(57)=spvae2e1(iv1)
      acd44(58)=abb44(17)
      acd44(59)=-acd44(2)*acd44(1)
      acd44(60)=-acd44(4)*acd44(3)
      acd44(61)=-acd44(6)*acd44(5)
      acd44(62)=-acd44(8)*acd44(7)
      acd44(63)=-acd44(10)*acd44(9)
      acd44(64)=-acd44(12)*acd44(11)
      acd44(65)=-acd44(14)*acd44(13)
      acd44(66)=-acd44(16)*acd44(15)
      acd44(67)=-acd44(18)*acd44(17)
      acd44(68)=-acd44(20)*acd44(19)
      acd44(69)=-acd44(22)*acd44(21)
      acd44(70)=-acd44(24)*acd44(23)
      acd44(71)=-acd44(26)*acd44(25)
      acd44(72)=-acd44(28)*acd44(27)
      acd44(73)=-acd44(30)*acd44(29)
      acd44(74)=-acd44(32)*acd44(31)
      acd44(75)=-acd44(34)*acd44(33)
      acd44(76)=-acd44(36)*acd44(35)
      acd44(77)=-acd44(38)*acd44(37)
      acd44(78)=-acd44(40)*acd44(39)
      acd44(79)=-acd44(42)*acd44(41)
      acd44(80)=-acd44(44)*acd44(43)
      acd44(81)=-acd44(46)*acd44(45)
      acd44(82)=-acd44(48)*acd44(47)
      acd44(83)=-acd44(50)*acd44(49)
      acd44(84)=-acd44(52)*acd44(51)
      acd44(85)=-acd44(54)*acd44(53)
      acd44(86)=-acd44(56)*acd44(55)
      acd44(87)=-acd44(58)*acd44(57)
      brack=acd44(59)+acd44(60)+acd44(61)+acd44(62)+acd44(63)+acd44(64)+acd44(6&
      &5)+acd44(66)+acd44(67)+acd44(68)+acd44(69)+acd44(70)+acd44(71)+acd44(72)&
      &+acd44(73)+acd44(74)+acd44(75)+acd44(76)+acd44(77)+acd44(78)+acd44(79)+a&
      &cd44(80)+acd44(81)+acd44(82)+acd44(83)+acd44(84)+acd44(85)+acd44(86)+acd&
      &44(87)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd44h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd44
      complex(ki) :: brack
      brack=0.0_ki
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd44h12_qp
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
      qshift = 0
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
end module     p2_gg_httbar_d44h12l1d_qp
