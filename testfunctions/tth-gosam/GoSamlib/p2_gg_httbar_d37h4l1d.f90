module     p2_gg_httbar_d37h4l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d37h4l1d.f90
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
      use p2_gg_httbar_abbrevd37h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd37
      complex(ki) :: brack
      acd37(1)=abb37(16)
      brack=acd37(1)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd37h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(60) :: acd37
      complex(ki) :: brack
      acd37(1)=k2(iv1)
      acd37(2)=abb37(18)
      acd37(3)=l3(iv1)
      acd37(4)=abb37(32)
      acd37(5)=l5(iv1)
      acd37(6)=abb37(50)
      acd37(7)=spvak1k2(iv1)
      acd37(8)=abb37(15)
      acd37(9)=spvak1l3(iv1)
      acd37(10)=abb37(36)
      acd37(11)=spvak2l3(iv1)
      acd37(12)=abb37(35)
      acd37(13)=spval3k1(iv1)
      acd37(14)=abb37(31)
      acd37(15)=spval3k2(iv1)
      acd37(16)=abb37(24)
      acd37(17)=spval3l5(iv1)
      acd37(18)=abb37(33)
      acd37(19)=spval5k1(iv1)
      acd37(20)=abb37(30)
      acd37(21)=spval5k2(iv1)
      acd37(22)=abb37(25)
      acd37(23)=spval5l3(iv1)
      acd37(24)=abb37(20)
      acd37(25)=spvae1k2(iv1)
      acd37(26)=abb37(17)
      acd37(27)=spvae2k2(iv1)
      acd37(28)=abb37(29)
      acd37(29)=spval3e1(iv1)
      acd37(30)=abb37(34)
      acd37(31)=spvae1l3(iv1)
      acd37(32)=abb37(28)
      acd37(33)=spval3e2(iv1)
      acd37(34)=abb37(19)
      acd37(35)=spvae2l3(iv1)
      acd37(36)=abb37(27)
      acd37(37)=spval5e1(iv1)
      acd37(38)=abb37(23)
      acd37(39)=spval5e2(iv1)
      acd37(40)=abb37(22)
      acd37(41)=acd37(2)*acd37(1)
      acd37(42)=acd37(4)*acd37(3)
      acd37(43)=acd37(6)*acd37(5)
      acd37(44)=acd37(8)*acd37(7)
      acd37(45)=acd37(10)*acd37(9)
      acd37(46)=acd37(12)*acd37(11)
      acd37(47)=acd37(14)*acd37(13)
      acd37(48)=acd37(16)*acd37(15)
      acd37(49)=acd37(18)*acd37(17)
      acd37(50)=acd37(20)*acd37(19)
      acd37(51)=acd37(22)*acd37(21)
      acd37(52)=acd37(24)*acd37(23)
      acd37(53)=acd37(26)*acd37(25)
      acd37(54)=acd37(28)*acd37(27)
      acd37(55)=acd37(30)*acd37(29)
      acd37(56)=acd37(32)*acd37(31)
      acd37(57)=acd37(34)*acd37(33)
      acd37(58)=acd37(36)*acd37(35)
      acd37(59)=acd37(38)*acd37(37)
      acd37(60)=acd37(40)*acd37(39)
      brack=acd37(41)+acd37(42)+acd37(43)+acd37(44)+acd37(45)+acd37(46)+acd37(4&
      &7)+acd37(48)+acd37(49)+acd37(50)+acd37(51)+acd37(52)+acd37(53)+acd37(54)&
      &+acd37(55)+acd37(56)+acd37(57)+acd37(58)+acd37(59)+acd37(60)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd37h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd37
      complex(ki) :: brack
      brack=0.0_ki
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd37h4
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
end module     p2_gg_httbar_d37h4l1d
