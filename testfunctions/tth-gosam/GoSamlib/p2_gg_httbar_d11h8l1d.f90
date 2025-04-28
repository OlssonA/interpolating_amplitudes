module     p2_gg_httbar_d11h8l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d11h8l1d.f90
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
      use p2_gg_httbar_abbrevd11h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd11
      complex(ki) :: brack
      acd11(1)=abb11(14)
      brack=acd11(1)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd11h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(51) :: acd11
      complex(ki) :: brack
      acd11(1)=k1(iv1)
      acd11(2)=abb11(11)
      acd11(3)=k2(iv1)
      acd11(4)=abb11(9)
      acd11(5)=l5(iv1)
      acd11(6)=abb11(17)
      acd11(7)=spvak1k2(iv1)
      acd11(8)=abb11(12)
      acd11(9)=spvak1l3(iv1)
      acd11(10)=abb11(18)
      acd11(11)=spvak1l5(iv1)
      acd11(12)=abb11(13)
      acd11(13)=spvak2k1(iv1)
      acd11(14)=abb11(26)
      acd11(15)=spvak2l3(iv1)
      acd11(16)=abb11(28)
      acd11(17)=spvak2l5(iv1)
      acd11(18)=abb11(10)
      acd11(19)=spval3k1(iv1)
      acd11(20)=abb11(23)
      acd11(21)=spval3k2(iv1)
      acd11(22)=abb11(21)
      acd11(23)=spval3l5(iv1)
      acd11(24)=abb11(39)
      acd11(25)=spval4k1(iv1)
      acd11(26)=abb11(30)
      acd11(27)=spval4k2(iv1)
      acd11(28)=abb11(38)
      acd11(29)=spval4l5(iv1)
      acd11(30)=abb11(20)
      acd11(31)=spval5k2(iv1)
      acd11(32)=abb11(36)
      acd11(33)=spval5l3(iv1)
      acd11(34)=abb11(22)
      acd11(35)=-acd11(2)*acd11(1)
      acd11(36)=-acd11(4)*acd11(3)
      acd11(37)=-acd11(6)*acd11(5)
      acd11(38)=-acd11(8)*acd11(7)
      acd11(39)=-acd11(10)*acd11(9)
      acd11(40)=-acd11(12)*acd11(11)
      acd11(41)=-acd11(14)*acd11(13)
      acd11(42)=-acd11(16)*acd11(15)
      acd11(43)=-acd11(18)*acd11(17)
      acd11(44)=-acd11(20)*acd11(19)
      acd11(45)=-acd11(22)*acd11(21)
      acd11(46)=-acd11(24)*acd11(23)
      acd11(47)=-acd11(26)*acd11(25)
      acd11(48)=-acd11(28)*acd11(27)
      acd11(49)=-acd11(30)*acd11(29)
      acd11(50)=-acd11(32)*acd11(31)
      acd11(51)=-acd11(34)*acd11(33)
      brack=acd11(35)+acd11(36)+acd11(37)+acd11(38)+acd11(39)+acd11(40)+acd11(4&
      &1)+acd11(42)+acd11(43)+acd11(44)+acd11(45)+acd11(46)+acd11(47)+acd11(48)&
      &+acd11(49)+acd11(50)+acd11(51)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd11h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(20) :: acd11
      complex(ki) :: brack
      acd11(1)=d(iv1,iv2)
      acd11(2)=abb11(15)
      acd11(3)=k1(iv1)
      acd11(4)=k2(iv2)
      acd11(5)=abb11(34)
      acd11(6)=spvak2l3(iv2)
      acd11(7)=abb11(25)
      acd11(8)=spval3l5(iv2)
      acd11(9)=abb11(19)
      acd11(10)=spval4l5(iv2)
      acd11(11)=abb11(16)
      acd11(12)=k1(iv2)
      acd11(13)=k2(iv1)
      acd11(14)=spvak2l3(iv1)
      acd11(15)=spval3l5(iv1)
      acd11(16)=spval4l5(iv1)
      acd11(17)=-acd11(14)*acd11(7)
      acd11(18)=acd11(15)*acd11(9)
      acd11(19)=-acd11(16)*acd11(11)
      acd11(17)=acd11(19)+acd11(18)+acd11(17)
      acd11(18)=acd11(12)-acd11(4)
      acd11(17)=acd11(18)*acd11(17)
      acd11(18)=-acd11(6)*acd11(7)
      acd11(19)=acd11(8)*acd11(9)
      acd11(20)=-acd11(10)*acd11(11)
      acd11(18)=acd11(20)+acd11(19)+acd11(18)
      acd11(19)=acd11(3)-acd11(13)
      acd11(18)=acd11(19)*acd11(18)
      acd11(19)=-acd11(12)+2.0_ki*acd11(4)
      acd11(19)=acd11(13)*acd11(19)
      acd11(20)=-acd11(3)*acd11(4)
      acd11(19)=acd11(20)+acd11(19)
      acd11(19)=acd11(5)*acd11(19)
      acd11(20)=acd11(2)*acd11(1)
      brack=acd11(17)+acd11(18)+acd11(19)+2.0_ki*acd11(20)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd11h8
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
end module     p2_gg_httbar_d11h8l1d
