module     p0_gg_gh_d11h3l1d_qp
   ! file: /mt/home/sjones/repos/POWHEG-BOX/ggh-gosam/GoSam_POWHEG/Virtual/p0_g &
   ! &g_gh/helicity3d11h3l1d_qp.f90
   ! generator: buildfortran_d.py
   use p0_gg_gh_config, only: ki => ki_qp
   use p0_gg_gh_util_qp, only: cond, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, private :: iv0
   integer, private :: iv1
   integer, private :: iv2
   integer, private :: iv3
   integer, private :: iv4
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd11h3_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(37) :: acd11
      complex(ki) :: brack
      acd11(1)=dotproduct(k1,qshift)
      acd11(2)=dotproduct(qshift,spvak2k3)
      acd11(3)=abb11(9)
      acd11(4)=dotproduct(k2,qshift)
      acd11(5)=abb11(21)
      acd11(6)=abb11(20)
      acd11(7)=dotproduct(qshift,spvak2k1)
      acd11(8)=abb11(12)
      acd11(9)=dotproduct(qshift,spvak2l4)
      acd11(10)=abb11(26)
      acd11(11)=dotproduct(qshift,spvak3k1)
      acd11(12)=abb11(27)
      acd11(13)=abb11(16)
      acd11(14)=dotproduct(k3,qshift)
      acd11(15)=abb11(19)
      acd11(16)=dotproduct(qshift,qshift)
      acd11(17)=abb11(29)
      acd11(18)=abb11(24)
      acd11(19)=dotproduct(qshift,spvak3k2)
      acd11(20)=abb11(7)
      acd11(21)=dotproduct(qshift,spval4k1)
      acd11(22)=abb11(23)
      acd11(23)=abb11(14)
      acd11(24)=abb11(15)
      acd11(25)=abb11(11)
      acd11(26)=abb11(13)
      acd11(27)=abb11(8)
      acd11(28)=abb11(17)
      acd11(29)=abb11(25)
      acd11(30)=abb11(10)
      acd11(31)=acd11(14)*acd11(15)
      acd11(32)=-acd11(16)*acd11(17)
      acd11(33)=acd11(9)*acd11(25)
      acd11(34)=acd11(11)*acd11(26)
      acd11(35)=acd11(4)*acd11(8)
      acd11(36)=-acd11(2)*acd11(19)*acd11(20)
      acd11(37)=acd11(7)*acd11(24)
      acd11(31)=acd11(37)+acd11(36)+acd11(35)+acd11(34)+acd11(33)+acd11(32)-acd&
      &11(27)+acd11(31)
      acd11(31)=acd11(7)*acd11(31)
      acd11(32)=acd11(21)*acd11(22)
      acd11(33)=acd11(1)*acd11(3)
      acd11(34)=-acd11(11)*acd11(17)
      acd11(35)=acd11(4)*acd11(6)
      acd11(32)=acd11(35)+acd11(34)+acd11(33)-acd11(23)+acd11(32)
      acd11(32)=acd11(2)*acd11(32)
      acd11(33)=acd11(9)*acd11(10)
      acd11(34)=acd11(11)*acd11(12)
      acd11(35)=acd11(4)*acd11(5)
      acd11(33)=acd11(35)+acd11(34)-acd11(13)+acd11(33)
      acd11(33)=acd11(4)*acd11(33)
      acd11(34)=acd11(16)*acd11(18)
      acd11(35)=-acd11(9)*acd11(28)
      acd11(36)=-acd11(11)*acd11(29)
      brack=acd11(30)+acd11(31)+acd11(32)+acd11(33)+acd11(34)+acd11(35)+acd11(3&
      &6)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd11h3_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(49) :: acd11
      complex(ki) :: brack
      acd11(1)=k1(iv1)
      acd11(2)=dotproduct(qshift,spvak2k3)
      acd11(3)=abb11(9)
      acd11(4)=k2(iv1)
      acd11(5)=dotproduct(k2,qshift)
      acd11(6)=abb11(21)
      acd11(7)=abb11(20)
      acd11(8)=dotproduct(qshift,spvak2k1)
      acd11(9)=abb11(12)
      acd11(10)=dotproduct(qshift,spvak2l4)
      acd11(11)=abb11(26)
      acd11(12)=dotproduct(qshift,spvak3k1)
      acd11(13)=abb11(27)
      acd11(14)=abb11(16)
      acd11(15)=k3(iv1)
      acd11(16)=abb11(19)
      acd11(17)=qshift(iv1)
      acd11(18)=abb11(29)
      acd11(19)=abb11(24)
      acd11(20)=spvak2k3(iv1)
      acd11(21)=dotproduct(k1,qshift)
      acd11(22)=dotproduct(qshift,spvak3k2)
      acd11(23)=abb11(7)
      acd11(24)=dotproduct(qshift,spval4k1)
      acd11(25)=abb11(23)
      acd11(26)=abb11(14)
      acd11(27)=spvak2k1(iv1)
      acd11(28)=dotproduct(k3,qshift)
      acd11(29)=dotproduct(qshift,qshift)
      acd11(30)=abb11(15)
      acd11(31)=abb11(11)
      acd11(32)=abb11(13)
      acd11(33)=abb11(8)
      acd11(34)=spvak2l4(iv1)
      acd11(35)=abb11(17)
      acd11(36)=spvak3k1(iv1)
      acd11(37)=abb11(25)
      acd11(38)=spvak3k2(iv1)
      acd11(39)=spval4k1(iv1)
      acd11(40)=acd11(16)*acd11(15)
      acd11(41)=acd11(34)*acd11(31)
      acd11(42)=acd11(36)*acd11(32)
      acd11(43)=2.0_ki*acd11(17)
      acd11(44)=-acd11(18)*acd11(43)
      acd11(45)=-acd11(20)*acd11(23)*acd11(22)
      acd11(46)=acd11(4)*acd11(9)
      acd11(47)=acd11(2)*acd11(23)
      acd11(48)=-acd11(38)*acd11(47)
      acd11(49)=acd11(27)*acd11(30)
      acd11(40)=2.0_ki*acd11(49)+acd11(48)+acd11(46)+acd11(45)+acd11(44)+acd11(&
      &42)+acd11(40)+acd11(41)
      acd11(40)=acd11(8)*acd11(40)
      acd11(41)=acd11(16)*acd11(28)
      acd11(42)=acd11(10)*acd11(31)
      acd11(44)=acd11(12)*acd11(32)
      acd11(45)=-acd11(18)*acd11(29)
      acd11(46)=acd11(5)*acd11(9)
      acd11(47)=-acd11(22)*acd11(47)
      acd11(41)=acd11(47)+acd11(46)+acd11(45)+acd11(44)+acd11(42)-acd11(33)+acd&
      &11(41)
      acd11(41)=acd11(27)*acd11(41)
      acd11(42)=acd11(25)*acd11(24)
      acd11(44)=acd11(3)*acd11(21)
      acd11(45)=-acd11(18)*acd11(12)
      acd11(46)=acd11(5)*acd11(7)
      acd11(42)=acd11(46)+acd11(45)+acd11(44)-acd11(26)+acd11(42)
      acd11(42)=acd11(20)*acd11(42)
      acd11(44)=acd11(25)*acd11(39)
      acd11(45)=acd11(3)*acd11(1)
      acd11(46)=-acd11(18)*acd11(36)
      acd11(47)=acd11(4)*acd11(7)
      acd11(44)=acd11(47)+acd11(46)+acd11(44)+acd11(45)
      acd11(44)=acd11(2)*acd11(44)
      acd11(45)=acd11(10)*acd11(11)
      acd11(46)=acd11(12)*acd11(13)
      acd11(47)=acd11(5)*acd11(6)
      acd11(45)=2.0_ki*acd11(47)+acd11(46)-acd11(14)+acd11(45)
      acd11(45)=acd11(4)*acd11(45)
      acd11(46)=acd11(34)*acd11(11)
      acd11(47)=acd11(36)*acd11(13)
      acd11(46)=acd11(46)+acd11(47)
      acd11(46)=acd11(5)*acd11(46)
      acd11(43)=acd11(19)*acd11(43)
      acd11(47)=-acd11(34)*acd11(35)
      acd11(48)=-acd11(36)*acd11(37)
      brack=acd11(40)+acd11(41)+acd11(42)+acd11(43)+acd11(44)+acd11(45)+acd11(4&
      &6)+acd11(47)+acd11(48)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd11h3_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(49) :: acd11
      complex(ki) :: brack
      acd11(1)=d(iv1,iv2)
      acd11(2)=dotproduct(qshift,spvak2k1)
      acd11(3)=abb11(29)
      acd11(4)=abb11(24)
      acd11(5)=k1(iv1)
      acd11(6)=spvak2k3(iv2)
      acd11(7)=abb11(9)
      acd11(8)=k1(iv2)
      acd11(9)=spvak2k3(iv1)
      acd11(10)=k2(iv1)
      acd11(11)=k2(iv2)
      acd11(12)=abb11(21)
      acd11(13)=spvak2k1(iv2)
      acd11(14)=abb11(12)
      acd11(15)=abb11(20)
      acd11(16)=spvak2l4(iv2)
      acd11(17)=abb11(26)
      acd11(18)=spvak3k1(iv2)
      acd11(19)=abb11(27)
      acd11(20)=spvak2k1(iv1)
      acd11(21)=spvak2l4(iv1)
      acd11(22)=spvak3k1(iv1)
      acd11(23)=k3(iv1)
      acd11(24)=abb11(19)
      acd11(25)=k3(iv2)
      acd11(26)=qshift(iv1)
      acd11(27)=qshift(iv2)
      acd11(28)=abb11(15)
      acd11(29)=dotproduct(qshift,spvak3k2)
      acd11(30)=abb11(7)
      acd11(31)=abb11(11)
      acd11(32)=abb11(13)
      acd11(33)=spvak3k2(iv2)
      acd11(34)=dotproduct(qshift,spvak2k3)
      acd11(35)=spvak3k2(iv1)
      acd11(36)=spval4k1(iv2)
      acd11(37)=abb11(23)
      acd11(38)=spval4k1(iv1)
      acd11(39)=acd11(24)*acd11(23)
      acd11(40)=acd11(21)*acd11(31)
      acd11(41)=acd11(22)*acd11(32)
      acd11(42)=acd11(10)*acd11(14)
      acd11(43)=2.0_ki*acd11(3)
      acd11(44)=-acd11(26)*acd11(43)
      acd11(45)=acd11(30)*acd11(34)
      acd11(46)=-acd11(35)*acd11(45)
      acd11(47)=acd11(30)*acd11(29)
      acd11(48)=-acd11(9)*acd11(47)
      acd11(49)=acd11(20)*acd11(28)
      acd11(39)=2.0_ki*acd11(49)+acd11(48)+acd11(46)+acd11(44)+acd11(42)+acd11(&
      &41)+acd11(39)+acd11(40)
      acd11(39)=acd11(13)*acd11(39)
      acd11(40)=acd11(24)*acd11(25)
      acd11(41)=acd11(16)*acd11(31)
      acd11(42)=acd11(18)*acd11(32)
      acd11(44)=acd11(11)*acd11(14)
      acd11(46)=-acd11(27)*acd11(43)
      acd11(45)=-acd11(33)*acd11(45)
      acd11(47)=-acd11(6)*acd11(47)
      acd11(40)=acd11(47)+acd11(45)+acd11(46)+acd11(44)+acd11(42)+acd11(40)+acd&
      &11(41)
      acd11(40)=acd11(20)*acd11(40)
      acd11(41)=acd11(37)*acd11(36)
      acd11(42)=acd11(7)*acd11(8)
      acd11(44)=acd11(11)*acd11(15)
      acd11(45)=-acd11(3)*acd11(18)
      acd11(46)=acd11(30)*acd11(2)
      acd11(47)=-acd11(33)*acd11(46)
      acd11(41)=acd11(47)+acd11(45)+acd11(44)+acd11(41)+acd11(42)
      acd11(41)=acd11(9)*acd11(41)
      acd11(42)=acd11(37)*acd11(38)
      acd11(44)=acd11(7)*acd11(5)
      acd11(45)=acd11(10)*acd11(15)
      acd11(47)=-acd11(3)*acd11(22)
      acd11(46)=-acd11(35)*acd11(46)
      acd11(42)=acd11(46)+acd11(47)+acd11(45)+acd11(42)+acd11(44)
      acd11(42)=acd11(6)*acd11(42)
      acd11(44)=acd11(16)*acd11(17)
      acd11(45)=acd11(18)*acd11(19)
      acd11(46)=acd11(11)*acd11(12)
      acd11(44)=2.0_ki*acd11(46)+acd11(44)+acd11(45)
      acd11(44)=acd11(10)*acd11(44)
      acd11(45)=acd11(17)*acd11(21)
      acd11(46)=acd11(22)*acd11(19)
      acd11(45)=acd11(45)+acd11(46)
      acd11(45)=acd11(11)*acd11(45)
      acd11(46)=2.0_ki*acd11(1)
      acd11(46)=acd11(4)*acd11(46)
      acd11(43)=-acd11(2)*acd11(1)*acd11(43)
      brack=acd11(39)+acd11(40)+acd11(41)+acd11(42)+acd11(43)+acd11(44)+acd11(4&
      &5)+acd11(46)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd11h3_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(18) :: acd11
      complex(ki) :: brack
      acd11(1)=d(iv1,iv2)
      acd11(2)=spvak2k1(iv3)
      acd11(3)=abb11(29)
      acd11(4)=d(iv1,iv3)
      acd11(5)=spvak2k1(iv2)
      acd11(6)=d(iv2,iv3)
      acd11(7)=spvak2k1(iv1)
      acd11(8)=spvak2k3(iv2)
      acd11(9)=spvak3k2(iv3)
      acd11(10)=abb11(7)
      acd11(11)=spvak2k3(iv3)
      acd11(12)=spvak3k2(iv2)
      acd11(13)=spvak2k3(iv1)
      acd11(14)=spvak3k2(iv1)
      acd11(15)=-acd11(11)*acd11(12)
      acd11(16)=-acd11(8)*acd11(9)
      acd11(15)=acd11(15)+acd11(16)
      acd11(15)=acd11(7)*acd11(15)
      acd11(16)=-acd11(11)*acd11(14)
      acd11(17)=-acd11(9)*acd11(13)
      acd11(16)=acd11(16)+acd11(17)
      acd11(16)=acd11(5)*acd11(16)
      acd11(17)=-acd11(12)*acd11(13)
      acd11(18)=-acd11(8)*acd11(14)
      acd11(17)=acd11(17)+acd11(18)
      acd11(17)=acd11(2)*acd11(17)
      acd11(15)=acd11(17)+acd11(15)+acd11(16)
      acd11(15)=acd11(10)*acd11(15)
      acd11(16)=-acd11(7)*acd11(6)
      acd11(17)=-acd11(5)*acd11(4)
      acd11(18)=-acd11(2)*acd11(1)
      acd11(16)=acd11(18)+acd11(16)+acd11(17)
      acd11(16)=acd11(3)*acd11(16)
      brack=acd11(15)+2.0_ki*acd11(16)
   end function brack_4
!---#] function brack_4:
!---#[ function brack_5:
   pure function brack_5(Q, mu2) result(brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd11h3_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd11
      complex(ki) :: brack
      brack=0.0_ki
   end function brack_5
!---#] function brack_5:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3,i4) result(numerator)
      use p0_gg_gh_globalsl1_qp, only: epspow
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_abbrevd11h3_qp
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      integer, intent(in), optional :: i2
      integer, intent(in), optional :: i3
      integer, intent(in), optional :: i4
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = -k3
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
      if(present(i4)) then
          iv4=i4
          deg=4
      else
          iv4=1
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
      if(deg.eq.4) then
         numerator = cond(epspow.eq.t1,brack_5,Q,mu2)
         return
      end if
   end function derivative
!---#] function derivative:
end module     p0_gg_gh_d11h3l1d_qp
