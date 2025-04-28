module     p0_gg_gh_d9h0l1d_qp
   ! file: /mt/home/sjones/repos/POWHEG-BOX/ggh-gosam/GoSam_POWHEG/Virtual/p0_g &
   ! &g_gh/helicity0d9h0l1d_qp.f90
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
      use p0_gg_gh_abbrevd9h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(22) :: acd9
      complex(ki) :: brack
      acd9(1)=dotproduct(k1,qshift)
      acd9(2)=dotproduct(qshift,spvak2k3)
      acd9(3)=abb9(12)
      acd9(4)=dotproduct(k2,qshift)
      acd9(5)=abb9(11)
      acd9(6)=dotproduct(qshift,qshift)
      acd9(7)=dotproduct(qshift,spvak1k2)
      acd9(8)=abb9(10)
      acd9(9)=abb9(13)
      acd9(10)=dotproduct(qshift,spvak1k3)
      acd9(11)=abb9(15)
      acd9(12)=dotproduct(qshift,spvak1l4)
      acd9(13)=abb9(8)
      acd9(14)=dotproduct(qshift,spval4k2)
      acd9(15)=abb9(14)
      acd9(16)=abb9(9)
      acd9(17)=-acd9(2)*acd9(8)
      acd9(17)=acd9(17)+acd9(9)
      acd9(17)=acd9(7)*acd9(17)
      acd9(18)=acd9(14)*acd9(15)
      acd9(19)=acd9(12)*acd9(13)
      acd9(20)=acd9(10)*acd9(11)
      acd9(21)=acd9(4)*acd9(5)
      acd9(22)=acd9(6)-acd9(1)
      acd9(22)=acd9(3)*acd9(22)
      acd9(17)=acd9(22)+acd9(21)+acd9(20)+acd9(19)-acd9(16)+acd9(18)+acd9(17)
      brack=acd9(17)*acd9(2)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd9h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(31) :: acd9
      complex(ki) :: brack
      acd9(1)=k1(iv1)
      acd9(2)=dotproduct(qshift,spvak2k3)
      acd9(3)=abb9(12)
      acd9(4)=k2(iv1)
      acd9(5)=abb9(11)
      acd9(6)=qshift(iv1)
      acd9(7)=spvak2k3(iv1)
      acd9(8)=dotproduct(k1,qshift)
      acd9(9)=dotproduct(k2,qshift)
      acd9(10)=dotproduct(qshift,qshift)
      acd9(11)=dotproduct(qshift,spvak1k2)
      acd9(12)=abb9(10)
      acd9(13)=abb9(13)
      acd9(14)=dotproduct(qshift,spvak1k3)
      acd9(15)=abb9(15)
      acd9(16)=dotproduct(qshift,spvak1l4)
      acd9(17)=abb9(8)
      acd9(18)=dotproduct(qshift,spval4k2)
      acd9(19)=abb9(14)
      acd9(20)=abb9(9)
      acd9(21)=spvak1k2(iv1)
      acd9(22)=spvak1k3(iv1)
      acd9(23)=spvak1l4(iv1)
      acd9(24)=spval4k2(iv1)
      acd9(25)=acd9(7)*acd9(11)
      acd9(26)=acd9(2)*acd9(21)
      acd9(25)=acd9(26)+2.0_ki*acd9(25)
      acd9(25)=acd9(12)*acd9(25)
      acd9(26)=-acd9(19)*acd9(24)
      acd9(27)=-acd9(17)*acd9(23)
      acd9(28)=-acd9(15)*acd9(22)
      acd9(29)=-acd9(13)*acd9(21)
      acd9(30)=-acd9(5)*acd9(4)
      acd9(31)=-2.0_ki*acd9(6)+acd9(1)
      acd9(31)=acd9(3)*acd9(31)
      acd9(25)=acd9(31)+acd9(30)+acd9(29)+acd9(28)+acd9(26)+acd9(27)+acd9(25)
      acd9(25)=acd9(2)*acd9(25)
      acd9(26)=-acd9(19)*acd9(18)
      acd9(27)=-acd9(17)*acd9(16)
      acd9(28)=-acd9(15)*acd9(14)
      acd9(29)=-acd9(11)*acd9(13)
      acd9(30)=-acd9(5)*acd9(9)
      acd9(31)=-acd9(10)+acd9(8)
      acd9(31)=acd9(3)*acd9(31)
      acd9(26)=acd9(31)+acd9(30)+acd9(29)+acd9(28)+acd9(27)+acd9(20)+acd9(26)
      acd9(26)=acd9(7)*acd9(26)
      brack=acd9(25)+acd9(26)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd9h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(35) :: acd9
      complex(ki) :: brack
      acd9(1)=d(iv1,iv2)
      acd9(2)=dotproduct(qshift,spvak2k3)
      acd9(3)=abb9(12)
      acd9(4)=k1(iv1)
      acd9(5)=spvak2k3(iv2)
      acd9(6)=k1(iv2)
      acd9(7)=spvak2k3(iv1)
      acd9(8)=k2(iv1)
      acd9(9)=abb9(11)
      acd9(10)=k2(iv2)
      acd9(11)=qshift(iv1)
      acd9(12)=qshift(iv2)
      acd9(13)=dotproduct(qshift,spvak1k2)
      acd9(14)=abb9(10)
      acd9(15)=spvak1k2(iv2)
      acd9(16)=abb9(13)
      acd9(17)=spvak1k3(iv2)
      acd9(18)=abb9(15)
      acd9(19)=spvak1l4(iv2)
      acd9(20)=abb9(8)
      acd9(21)=spval4k2(iv2)
      acd9(22)=abb9(14)
      acd9(23)=spvak1k2(iv1)
      acd9(24)=spvak1k3(iv1)
      acd9(25)=spvak1l4(iv1)
      acd9(26)=spval4k2(iv1)
      acd9(27)=2.0_ki*acd9(2)
      acd9(28)=acd9(27)*acd9(14)
      acd9(28)=acd9(28)-acd9(16)
      acd9(29)=-acd9(23)*acd9(28)
      acd9(30)=acd9(22)*acd9(26)
      acd9(31)=acd9(20)*acd9(25)
      acd9(32)=acd9(18)*acd9(24)
      acd9(33)=acd9(9)*acd9(8)
      acd9(34)=2.0_ki*acd9(11)-acd9(4)
      acd9(34)=acd9(3)*acd9(34)
      acd9(35)=2.0_ki*acd9(14)
      acd9(35)=-acd9(7)*acd9(13)*acd9(35)
      acd9(29)=acd9(35)+acd9(34)+acd9(33)+acd9(32)+acd9(30)+acd9(31)+acd9(29)
      acd9(29)=acd9(5)*acd9(29)
      acd9(28)=-acd9(15)*acd9(28)
      acd9(30)=acd9(22)*acd9(21)
      acd9(31)=acd9(20)*acd9(19)
      acd9(32)=acd9(18)*acd9(17)
      acd9(33)=acd9(9)*acd9(10)
      acd9(34)=2.0_ki*acd9(12)-acd9(6)
      acd9(34)=acd9(3)*acd9(34)
      acd9(28)=acd9(34)+acd9(33)+acd9(32)+acd9(30)+acd9(31)+acd9(28)
      acd9(28)=acd9(7)*acd9(28)
      acd9(27)=acd9(3)*acd9(1)*acd9(27)
      brack=acd9(27)+acd9(28)+acd9(29)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd9h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(14) :: acd9
      complex(ki) :: brack
      acd9(1)=d(iv1,iv2)
      acd9(2)=spvak2k3(iv3)
      acd9(3)=abb9(12)
      acd9(4)=d(iv1,iv3)
      acd9(5)=spvak2k3(iv2)
      acd9(6)=d(iv2,iv3)
      acd9(7)=spvak2k3(iv1)
      acd9(8)=spvak1k2(iv3)
      acd9(9)=abb9(10)
      acd9(10)=spvak1k2(iv2)
      acd9(11)=spvak1k2(iv1)
      acd9(12)=-acd9(1)*acd9(2)
      acd9(13)=-acd9(4)*acd9(5)
      acd9(14)=-acd9(6)*acd9(7)
      acd9(12)=acd9(14)+acd9(12)+acd9(13)
      acd9(12)=acd9(3)*acd9(12)
      acd9(13)=acd9(8)*acd9(5)
      acd9(14)=acd9(10)*acd9(2)
      acd9(13)=acd9(14)+acd9(13)
      acd9(13)=acd9(13)*acd9(7)
      acd9(14)=acd9(11)*acd9(2)*acd9(5)
      acd9(13)=acd9(14)+acd9(13)
      acd9(13)=acd9(9)*acd9(13)
      acd9(12)=acd9(12)+acd9(13)
      brack=2.0_ki*acd9(12)
   end function brack_4
!---#] function brack_4:
!---#[ function brack_5:
   pure function brack_5(Q, mu2) result(brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd9h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd9
      complex(ki) :: brack
      brack=0.0_ki
   end function brack_5
!---#] function brack_5:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3,i4) result(numerator)
      use p0_gg_gh_globalsl1_qp, only: epspow
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_abbrevd9h0_qp
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
end module     p0_gg_gh_d9h0l1d_qp
