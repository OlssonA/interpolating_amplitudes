module     p2_gg_httbar_d42h4l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d42h4l1d.f90
   ! generator: buildfortran_d.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd42h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd42
      complex(ki) :: brack
      acd42(1)=abb42(14)
      brack=acd42(1)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd42h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(87) :: acd42
      complex(ki) :: brack
      acd42(1)=k1(iv1)
      acd42(2)=abb42(23)
      acd42(3)=k2(iv1)
      acd42(4)=abb42(21)
      acd42(5)=l5(iv1)
      acd42(6)=abb42(36)
      acd42(7)=spvak1k2(iv1)
      acd42(8)=abb42(19)
      acd42(9)=spvak1l4(iv1)
      acd42(10)=abb42(29)
      acd42(11)=spvak1l5(iv1)
      acd42(12)=abb42(18)
      acd42(13)=spvak2k1(iv1)
      acd42(14)=abb42(15)
      acd42(15)=spvak2l4(iv1)
      acd42(16)=abb42(60)
      acd42(17)=spvak2l5(iv1)
      acd42(18)=abb42(49)
      acd42(19)=spval4k1(iv1)
      acd42(20)=abb42(16)
      acd42(21)=spval4l5(iv1)
      acd42(22)=abb42(37)
      acd42(23)=spval5k1(iv1)
      acd42(24)=abb42(30)
      acd42(25)=spval5k2(iv1)
      acd42(26)=abb42(57)
      acd42(27)=spval5l4(iv1)
      acd42(28)=abb42(55)
      acd42(29)=spvak1e1(iv1)
      acd42(30)=abb42(46)
      acd42(31)=spvae1k1(iv1)
      acd42(32)=abb42(40)
      acd42(33)=spvak1e2(iv1)
      acd42(34)=abb42(22)
      acd42(35)=spvae2k1(iv1)
      acd42(36)=abb42(67)
      acd42(37)=spvak2e1(iv1)
      acd42(38)=abb42(65)
      acd42(39)=spvae1k2(iv1)
      acd42(40)=abb42(45)
      acd42(41)=spvak2e2(iv1)
      acd42(42)=abb42(28)
      acd42(43)=spval4e1(iv1)
      acd42(44)=abb42(62)
      acd42(45)=spvae1l4(iv1)
      acd42(46)=abb42(59)
      acd42(47)=spval5e1(iv1)
      acd42(48)=abb42(24)
      acd42(49)=spvae1l5(iv1)
      acd42(50)=abb42(50)
      acd42(51)=spval5e2(iv1)
      acd42(52)=abb42(34)
      acd42(53)=spvae2l5(iv1)
      acd42(54)=abb42(26)
      acd42(55)=spvae1e2(iv1)
      acd42(56)=abb42(35)
      acd42(57)=spvae2e1(iv1)
      acd42(58)=abb42(31)
      acd42(59)=acd42(2)*acd42(1)
      acd42(60)=acd42(4)*acd42(3)
      acd42(61)=acd42(6)*acd42(5)
      acd42(62)=acd42(8)*acd42(7)
      acd42(63)=acd42(10)*acd42(9)
      acd42(64)=acd42(12)*acd42(11)
      acd42(65)=acd42(14)*acd42(13)
      acd42(66)=acd42(16)*acd42(15)
      acd42(67)=acd42(18)*acd42(17)
      acd42(68)=acd42(20)*acd42(19)
      acd42(69)=acd42(22)*acd42(21)
      acd42(70)=acd42(24)*acd42(23)
      acd42(71)=acd42(26)*acd42(25)
      acd42(72)=acd42(28)*acd42(27)
      acd42(73)=acd42(30)*acd42(29)
      acd42(74)=acd42(32)*acd42(31)
      acd42(75)=acd42(34)*acd42(33)
      acd42(76)=acd42(36)*acd42(35)
      acd42(77)=acd42(38)*acd42(37)
      acd42(78)=acd42(40)*acd42(39)
      acd42(79)=acd42(42)*acd42(41)
      acd42(80)=acd42(44)*acd42(43)
      acd42(81)=acd42(46)*acd42(45)
      acd42(82)=acd42(48)*acd42(47)
      acd42(83)=acd42(50)*acd42(49)
      acd42(84)=acd42(52)*acd42(51)
      acd42(85)=acd42(54)*acd42(53)
      acd42(86)=acd42(56)*acd42(55)
      acd42(87)=acd42(58)*acd42(57)
      brack=acd42(59)+acd42(60)+acd42(61)+acd42(62)+acd42(63)+acd42(64)+acd42(6&
      &5)+acd42(66)+acd42(67)+acd42(68)+acd42(69)+acd42(70)+acd42(71)+acd42(72)&
      &+acd42(73)+acd42(74)+acd42(75)+acd42(76)+acd42(77)+acd42(78)+acd42(79)+a&
      &cd42(80)+acd42(81)+acd42(82)+acd42(83)+acd42(84)+acd42(85)+acd42(86)+acd&
      &42(87)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd42h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd42
      complex(ki) :: brack
      brack=0.0_ki
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd42h4
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
end module     p2_gg_httbar_d42h4l1d
