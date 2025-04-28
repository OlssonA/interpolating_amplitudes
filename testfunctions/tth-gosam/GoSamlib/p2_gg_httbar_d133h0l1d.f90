module     p2_gg_httbar_d133h0l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d133h0l1d.f90
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
      use p2_gg_httbar_abbrevd133h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd133
      complex(ki) :: brack
      acd133(1)=abb133(40)
      brack=acd133(1)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd133h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(72) :: acd133
      complex(ki) :: brack
      acd133(1)=k2(iv1)
      acd133(2)=abb133(13)
      acd133(3)=l5(iv1)
      acd133(4)=abb133(38)
      acd133(5)=e2(iv1)
      acd133(6)=abb133(43)
      acd133(7)=spvak1l5(iv1)
      acd133(8)=abb133(20)
      acd133(9)=spvak2k1(iv1)
      acd133(10)=abb133(23)
      acd133(11)=spvak2l4(iv1)
      acd133(12)=abb133(19)
      acd133(13)=spvak2l5(iv1)
      acd133(14)=abb133(11)
      acd133(15)=spval4l5(iv1)
      acd133(16)=abb133(21)
      acd133(17)=spval5k1(iv1)
      acd133(18)=abb133(17)
      acd133(19)=spval5k2(iv1)
      acd133(20)=abb133(18)
      acd133(21)=spval5l4(iv1)
      acd133(22)=abb133(55)
      acd133(23)=spvak1e2(iv1)
      acd133(24)=abb133(50)
      acd133(25)=spvae2k1(iv1)
      acd133(26)=abb133(35)
      acd133(27)=spvak2e1(iv1)
      acd133(28)=abb133(39)
      acd133(29)=spvak2e2(iv1)
      acd133(30)=abb133(16)
      acd133(31)=spvae2k2(iv1)
      acd133(32)=abb133(14)
      acd133(33)=spval4e2(iv1)
      acd133(34)=abb133(125)
      acd133(35)=spvae2l4(iv1)
      acd133(36)=abb133(120)
      acd133(37)=spval5e1(iv1)
      acd133(38)=abb133(41)
      acd133(39)=spvae1l5(iv1)
      acd133(40)=abb133(26)
      acd133(41)=spval5e2(iv1)
      acd133(42)=abb133(46)
      acd133(43)=spvae2l5(iv1)
      acd133(44)=abb133(69)
      acd133(45)=spvae1e2(iv1)
      acd133(46)=abb133(47)
      acd133(47)=spvae2e1(iv1)
      acd133(48)=abb133(94)
      acd133(49)=acd133(2)*acd133(1)
      acd133(50)=acd133(4)*acd133(3)
      acd133(51)=acd133(6)*acd133(5)
      acd133(52)=acd133(8)*acd133(7)
      acd133(53)=acd133(10)*acd133(9)
      acd133(54)=acd133(12)*acd133(11)
      acd133(55)=acd133(14)*acd133(13)
      acd133(56)=acd133(16)*acd133(15)
      acd133(57)=acd133(18)*acd133(17)
      acd133(58)=acd133(20)*acd133(19)
      acd133(59)=acd133(22)*acd133(21)
      acd133(60)=acd133(24)*acd133(23)
      acd133(61)=acd133(26)*acd133(25)
      acd133(62)=acd133(28)*acd133(27)
      acd133(63)=acd133(30)*acd133(29)
      acd133(64)=acd133(32)*acd133(31)
      acd133(65)=acd133(34)*acd133(33)
      acd133(66)=acd133(36)*acd133(35)
      acd133(67)=acd133(38)*acd133(37)
      acd133(68)=acd133(40)*acd133(39)
      acd133(69)=acd133(42)*acd133(41)
      acd133(70)=acd133(44)*acd133(43)
      acd133(71)=acd133(46)*acd133(45)
      acd133(72)=acd133(48)*acd133(47)
      brack=acd133(49)+acd133(50)+acd133(51)+acd133(52)+acd133(53)+acd133(54)+a&
      &cd133(55)+acd133(56)+acd133(57)+acd133(58)+acd133(59)+acd133(60)+acd133(&
      &61)+acd133(62)+acd133(63)+acd133(64)+acd133(65)+acd133(66)+acd133(67)+ac&
      &d133(68)+acd133(69)+acd133(70)+acd133(71)+acd133(72)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd133h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(41) :: acd133
      complex(ki) :: brack
      acd133(1)=d(iv1,iv2)
      acd133(2)=abb133(105)
      acd133(3)=k2(iv1)
      acd133(4)=e2(iv2)
      acd133(5)=abb133(65)
      acd133(6)=k2(iv2)
      acd133(7)=e2(iv1)
      acd133(8)=l5(iv1)
      acd133(9)=abb133(57)
      acd133(10)=l5(iv2)
      acd133(11)=spvak1k2(iv2)
      acd133(12)=abb133(22)
      acd133(13)=spval4k2(iv2)
      acd133(14)=abb133(25)
      acd133(15)=spval5k1(iv2)
      acd133(16)=abb133(12)
      acd133(17)=spval5k2(iv2)
      acd133(18)=abb133(15)
      acd133(19)=spval5l4(iv2)
      acd133(20)=abb133(30)
      acd133(21)=spvae1k2(iv2)
      acd133(22)=abb133(24)
      acd133(23)=spval5e1(iv2)
      acd133(24)=abb133(86)
      acd133(25)=spvak1k2(iv1)
      acd133(26)=spval4k2(iv1)
      acd133(27)=spval5k1(iv1)
      acd133(28)=spval5k2(iv1)
      acd133(29)=spval5l4(iv1)
      acd133(30)=spvae1k2(iv1)
      acd133(31)=spval5e1(iv1)
      acd133(32)=acd133(3)*acd133(5)
      acd133(33)=acd133(8)*acd133(9)
      acd133(34)=acd133(25)*acd133(12)
      acd133(35)=acd133(26)*acd133(14)
      acd133(36)=acd133(27)*acd133(16)
      acd133(37)=acd133(28)*acd133(18)
      acd133(38)=acd133(29)*acd133(20)
      acd133(39)=acd133(30)*acd133(22)
      acd133(40)=acd133(31)*acd133(24)
      acd133(32)=acd133(40)+acd133(39)+acd133(38)+acd133(37)+acd133(36)+acd133(&
      &35)+acd133(34)+acd133(33)+acd133(32)
      acd133(32)=acd133(4)*acd133(32)
      acd133(33)=acd133(6)*acd133(5)
      acd133(34)=acd133(10)*acd133(9)
      acd133(35)=acd133(11)*acd133(12)
      acd133(36)=acd133(13)*acd133(14)
      acd133(37)=acd133(15)*acd133(16)
      acd133(38)=acd133(17)*acd133(18)
      acd133(39)=acd133(19)*acd133(20)
      acd133(40)=acd133(21)*acd133(22)
      acd133(41)=acd133(23)*acd133(24)
      acd133(33)=acd133(41)+acd133(40)+acd133(39)+acd133(38)+acd133(37)+acd133(&
      &36)+acd133(35)+acd133(34)+acd133(33)
      acd133(33)=acd133(7)*acd133(33)
      acd133(34)=acd133(2)*acd133(1)
      brack=acd133(32)+acd133(33)-2.0_ki*acd133(34)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd133h0
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
end module     p2_gg_httbar_d133h0l1d
