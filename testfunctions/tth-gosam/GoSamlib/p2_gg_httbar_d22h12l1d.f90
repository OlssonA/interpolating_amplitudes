module     p2_gg_httbar_d22h12l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d22h12l1d.f90
   ! generator: buildfortran_d.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd22h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(35) :: acd22
      complex(ki) :: brack
      acd22(1)=dotproduct(k2,qshift)
      acd22(2)=dotproduct(qshift,spvak2l4)
      acd22(3)=abb22(10)
      acd22(4)=dotproduct(qshift,spvak2l5)
      acd22(5)=abb22(20)
      acd22(6)=abb22(18)
      acd22(7)=dotproduct(qshift,qshift)
      acd22(8)=abb22(12)
      acd22(9)=abb22(7)
      acd22(10)=abb22(13)
      acd22(11)=dotproduct(qshift,spvak1l3)
      acd22(12)=abb22(8)
      acd22(13)=dotproduct(qshift,spvak1l4)
      acd22(14)=dotproduct(qshift,spvak2k1)
      acd22(15)=abb22(22)
      acd22(16)=abb22(17)
      acd22(17)=dotproduct(qshift,spvak1l5)
      acd22(18)=abb22(21)
      acd22(19)=abb22(14)
      acd22(20)=abb22(16)
      acd22(21)=dotproduct(qshift,spvak2l3)
      acd22(22)=abb22(15)
      acd22(23)=dotproduct(qshift,spval3k1)
      acd22(24)=abb22(19)
      acd22(25)=abb22(6)
      acd22(26)=acd22(3)*acd22(2)
      acd22(27)=acd22(5)*acd22(4)
      acd22(26)=-acd22(6)+acd22(26)+acd22(27)
      acd22(26)=acd22(1)*acd22(26)
      acd22(27)=acd22(15)*acd22(13)
      acd22(28)=acd22(18)*acd22(17)
      acd22(27)=-acd22(19)+acd22(28)+acd22(27)
      acd22(27)=acd22(14)*acd22(27)
      acd22(28)=acd22(8)*acd22(7)
      acd22(29)=-acd22(9)*acd22(2)
      acd22(30)=-acd22(10)*acd22(4)
      acd22(31)=-acd22(12)*acd22(11)
      acd22(32)=-acd22(16)*acd22(13)
      acd22(33)=-acd22(20)*acd22(17)
      acd22(34)=-acd22(22)*acd22(21)
      acd22(35)=-acd22(24)*acd22(23)
      brack=acd22(25)+acd22(26)+acd22(27)+acd22(28)+acd22(29)+acd22(30)+acd22(3&
      &1)+acd22(32)+acd22(33)+acd22(34)+acd22(35)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd22h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(40) :: acd22
      complex(ki) :: brack
      acd22(1)=k2(iv1)
      acd22(2)=dotproduct(qshift,spvak2l4)
      acd22(3)=abb22(10)
      acd22(4)=dotproduct(qshift,spvak2l5)
      acd22(5)=abb22(20)
      acd22(6)=abb22(18)
      acd22(7)=qshift(iv1)
      acd22(8)=abb22(12)
      acd22(9)=spvak2l4(iv1)
      acd22(10)=dotproduct(k2,qshift)
      acd22(11)=abb22(7)
      acd22(12)=spvak2l5(iv1)
      acd22(13)=abb22(13)
      acd22(14)=spvak1l3(iv1)
      acd22(15)=abb22(8)
      acd22(16)=spvak1l4(iv1)
      acd22(17)=dotproduct(qshift,spvak2k1)
      acd22(18)=abb22(22)
      acd22(19)=abb22(17)
      acd22(20)=spvak2k1(iv1)
      acd22(21)=dotproduct(qshift,spvak1l4)
      acd22(22)=dotproduct(qshift,spvak1l5)
      acd22(23)=abb22(21)
      acd22(24)=abb22(14)
      acd22(25)=spvak1l5(iv1)
      acd22(26)=abb22(16)
      acd22(27)=spvak2l3(iv1)
      acd22(28)=abb22(15)
      acd22(29)=spval3k1(iv1)
      acd22(30)=abb22(19)
      acd22(31)=acd22(2)*acd22(3)
      acd22(32)=acd22(4)*acd22(5)
      acd22(31)=-acd22(6)+acd22(32)+acd22(31)
      acd22(31)=acd22(1)*acd22(31)
      acd22(32)=acd22(21)*acd22(18)
      acd22(33)=acd22(22)*acd22(23)
      acd22(32)=-acd22(24)+acd22(33)+acd22(32)
      acd22(32)=acd22(20)*acd22(32)
      acd22(33)=acd22(10)*acd22(3)
      acd22(33)=-acd22(11)+acd22(33)
      acd22(33)=acd22(9)*acd22(33)
      acd22(34)=acd22(10)*acd22(5)
      acd22(34)=-acd22(13)+acd22(34)
      acd22(34)=acd22(12)*acd22(34)
      acd22(35)=acd22(18)*acd22(17)
      acd22(35)=-acd22(19)+acd22(35)
      acd22(35)=acd22(16)*acd22(35)
      acd22(36)=acd22(23)*acd22(17)
      acd22(36)=-acd22(26)+acd22(36)
      acd22(36)=acd22(25)*acd22(36)
      acd22(37)=acd22(8)*acd22(7)
      acd22(38)=-acd22(15)*acd22(14)
      acd22(39)=-acd22(28)*acd22(27)
      acd22(40)=-acd22(30)*acd22(29)
      brack=acd22(31)+acd22(32)+acd22(33)+acd22(34)+acd22(35)+acd22(36)+2.0_ki*&
      &acd22(37)+acd22(38)+acd22(39)+acd22(40)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd22h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(23) :: acd22
      complex(ki) :: brack
      acd22(1)=d(iv1,iv2)
      acd22(2)=abb22(12)
      acd22(3)=k2(iv1)
      acd22(4)=spvak2l4(iv2)
      acd22(5)=abb22(10)
      acd22(6)=spvak2l5(iv2)
      acd22(7)=abb22(20)
      acd22(8)=k2(iv2)
      acd22(9)=spvak2l4(iv1)
      acd22(10)=spvak2l5(iv1)
      acd22(11)=spvak1l4(iv1)
      acd22(12)=spvak2k1(iv2)
      acd22(13)=abb22(22)
      acd22(14)=spvak1l4(iv2)
      acd22(15)=spvak2k1(iv1)
      acd22(16)=spvak1l5(iv2)
      acd22(17)=abb22(21)
      acd22(18)=spvak1l5(iv1)
      acd22(19)=acd22(4)*acd22(5)
      acd22(20)=acd22(6)*acd22(7)
      acd22(19)=acd22(20)+acd22(19)
      acd22(19)=acd22(3)*acd22(19)
      acd22(20)=acd22(9)*acd22(5)
      acd22(21)=acd22(10)*acd22(7)
      acd22(20)=acd22(21)+acd22(20)
      acd22(20)=acd22(8)*acd22(20)
      acd22(21)=acd22(11)*acd22(12)
      acd22(22)=acd22(14)*acd22(15)
      acd22(21)=acd22(22)+acd22(21)
      acd22(21)=acd22(13)*acd22(21)
      acd22(22)=acd22(16)*acd22(15)
      acd22(23)=acd22(18)*acd22(12)
      acd22(22)=acd22(23)+acd22(22)
      acd22(22)=acd22(17)*acd22(22)
      acd22(23)=acd22(2)*acd22(1)
      brack=acd22(19)+acd22(20)+acd22(21)+acd22(22)+2.0_ki*acd22(23)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd22h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd22
      complex(ki) :: brack
      brack=0.0_ki
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd22h12
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
      qshift = k3+k5
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
end module     p2_gg_httbar_d22h12l1d
